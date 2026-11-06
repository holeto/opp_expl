// Validates slumbot_api against the live agent and times it.
//
//   validate <slumbot_dir> selfplay <n_hands> <seed> [extra_hands]
//     Random hands: Slumbot plays through HandleStateChange (the code path behind slumbot.com) with
//     a random hand number; the opponent picks random legal actions with arbitrary bet sizes, so
//     most hands go through translation. At every Slumbot decision:
//       - sb_query's distribution at u = u(hand number) must be pure and its action must equal the
//         live action, for Slumbot's hand and for extra_hands (default 8) other hands, each
//         compared with its own HandleStateChange call;
//       - the cells of the history must have masses summing to 1, the cell that contains the
//         true u must be feasible, and its point must give the same abstract path as the true u.
//   validate <slumbot_dir> bench <n_states> <seed>
//     Collects Slumbot decision states from self-play, then times Prepare + all 1326 hands per
//     state, sb_cells per state, and one HandleStateChange per hand for comparison.
//   validate <slumbot_dir> tree <sb|bb> <out_path>
//     sb_export_tree.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "cards.h"
#include "poker_action.h"
#include "slumbot_api.h"
#include "slumbot_api_impl.h"

using sbapi::Cell;
using sbapi::OutAction;
using sbapi::PublicState;
using sbapi::SlumbotApi;
using std::string;
using std::vector;

namespace {

constexpr int kSmallBlind = 50, kBigBlind = 100, kStack = 20000;
const int kBoardCards[4] = {0, 3, 4, 5};

string CardStr(int c) {
  const char *ranks = "23456789TJQKA", *suits = "cdhs";
  return string(1, ranks[c / 4]) + suits[c % 4];
}

string MatchState(int pos, int hand_no, const string &action, int c1, int c2, const int *board,
                  int street) {
  string hole = CardStr(c1) + CardStr(c2);
  string cards = pos == 0 ? hole + "|" : "|" + hole;
  if (street >= 1) cards += "/" + CardStr(board[0]) + CardStr(board[1]) + CardStr(board[2]);
  if (street >= 2) cards += "/" + CardStr(board[3]);
  if (street >= 3) cards += "/" + CardStr(board[4]);
  return "MATCHSTATE:" + std::to_string(pos) + ":" + std::to_string(hand_no) + ":" + action + ":" +
         cards;
}

struct State {
  int pos, st, street_last_bet_to, total_last_bet_to, last_bet_size, last_bettor;
};

bool GetState(const string &action, State *s) {
  ActionError error;
  return ActionState(action, s->pos, s->st, kSmallBlind, kBigBlind, kStack, s->street_last_bet_to,
                     s->total_last_bet_to, s->last_bet_size, s->last_bettor, error) &&
         error == ActionError::NO_ERROR;
}

// A random legal action with arbitrary sizes: pot fractions from small to overbets, min raises,
// all-ins and uniform fractions.
string RandomAction(const State &s, std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  bool facing = s.last_bet_size > 0;
  int remaining = kStack - s.total_last_bet_to;
  double x = unif(rng);
  if (facing && x < 0.15) return "f";
  bool can_bet = remaining > 0;
  if (!can_bet || x < 0.55) return facing ? "c" : "k";
  int min_bet = facing ? std::max(s.last_bet_size, kBigBlind) : kBigBlind;
  min_bet = std::min(min_bet, remaining);
  int pot = 2 * s.total_last_bet_to;  // pot after calling, roughly
  static const double kFracs[] = {0.1, 0.25, 0.33, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0};
  double y = unif(rng);
  int size;
  if (y < 0.1) {
    size = min_bet;
  } else if (y < 0.18) {
    size = remaining;
  } else if (y < 0.45) {
    size = static_cast<int>(std::lround((0.05 + 3.95 * unif(rng)) * pot));
  } else {
    size = static_cast<int>(std::lround(kFracs[rng() % 12] * pot));
  }
  size = std::max(min_bet, std::min(size, remaining));
  return "b" + std::to_string(s.street_last_bet_to + size);
}

string LiveIncr(BotAction ba, int we_bet_to) {
  switch (ba) {
    case BA_FOLD: return "f";
    case BA_CALL: return "c";
    case BA_CHECK: return "k";
    case BA_BET: return "b" + std::to_string(we_bet_to);
    default: return "";
  }
}

// The single action of a pure distribution as a protocol increment; "" if not pure.
string ApiIncr(const vector<OutAction> &out) {
  string incr;
  int n_one = 0;
  for (const OutAction &a : out) {
    if (a.prob == 1.0) {
      ++n_one;
      incr = a.type == SB_FOLD ? "f" : a.type == SB_CHECK ? "k" : a.type == SB_CALL ? "c"
                                                                                    : "b" + std::to_string(a.bet);
    } else if (a.prob != 0.0) {
      return "";
    }
  }
  return n_one == 1 ? incr : "";
}

struct Deal {
  int slumbot_pos, hand_no;
  int cards[9];  // Slumbot's two, the opponent's two, five board cards
};

Deal RandomDeal(std::mt19937_64 &rng) {
  Deal d;
  d.slumbot_pos = rng() % 2;
  d.hand_no = rng() % 1000000;
  vector<int> deck(52);
  for (int i = 0; i < 52; ++i) deck[i] = i;
  for (int i = 0; i < 9; ++i) {
    int j = i + rng() % (52 - i);
    std::swap(deck[i], deck[j]);
    d.cards[i] = deck[i];
  }
  return d;
}

// Other hands for a state: random pairs disjoint from the board.
vector<std::pair<int, int>> ExtraHands(const int *board, int street, int n, std::mt19937_64 &rng) {
  vector<std::pair<int, int>> hands;
  while ((int)hands.size() < n) {
    int c1 = rng() % 52, c2 = rng() % 52;
    bool ok = c1 != c2;
    for (int i = 0; i < kBoardCards[street]; ++i) ok = ok && board[i] != c1 && board[i] != c2;
    if (ok) hands.push_back({c1, c2});
  }
  return hands;
}

struct Stats {
  long hands = 0, decisions = 0, comparisons = 0, mismatches = 0, not_pure = 0;
  long live_none = 0, api_error_live_acted = 0, both_none = 0, opp_illegal = 0, live_illegal = 0;
  long batch_checks = 0;
  long cell_checks = 0, cell_failures = 0, decisions_translated = 0, decisions_stalled = 0;
  std::map<int, long> cells_hist, status_hist;
  int printed = 0;
};

void Report(Stats &st, const string &what, const string &ms, const string &live,
            const string &api) {
  ++st.mismatches;
  if (st.printed++ < 25) {
    printf("MISMATCH %s\n  %s\n  live=%s api=%s\n", what.c_str(), ms.c_str(), live.c_str(),
           api.c_str());
  }
}

// Plays one hand; calls on_decision(action, deal, street) at each Slumbot decision before the
// live action is applied. Returns false if the walk was cut short.
template <class F>
void PlayHand(SlumbotApi *api, const Deal &d, std::mt19937_64 &rng, Stats &st, F on_decision) {
  string action;
  const int *board = d.cards + 4;
  for (int step = 0; step < 200; ++step) {
    State s;
    if (!GetState(action, &s)) return;
    if (s.pos == -1) return;
    string incr;
    if (s.pos == d.slumbot_pos) {
      on_decision(action, s.st);
      int we_bet_to;
      string ms = MatchState(d.slumbot_pos, d.hand_no, action, d.cards[0], d.cards[1], board, s.st);
      BotAction ba = api->HandleStateChange(ms, &we_bet_to);
      incr = LiveIncr(ba, we_bet_to);
      if (incr.empty()) return;  // live agent answers nothing (counted in on_decision)
    } else {
      incr = RandomAction(s, rng);
    }
    string next;
    ActionError error;
    if (!ExtendAction(action, incr, kSmallBlind, kBigBlind, kStack, next, error)) {
      if (s.pos == d.slumbot_pos) ++st.live_illegal;
      else ++st.opp_illegal;
      return;
    }
    action = next;
  }
}

int SelfPlay(SlumbotApi *api, int n_hands, uint64_t seed, int n_extra) {
  std::mt19937_64 rng(seed);
  Stats st;
  for (int hand = 0; hand < n_hands; ++hand) {
    Deal d = RandomDeal(rng);
    ++st.hands;
    const int *board = d.cards + 4;
    double u = sb_u_from_hand_no(d.hand_no);
    PlayHand(api, d, rng, st, [&](const string &action, int street) {
      ++st.decisions;
      PublicState ps;
      int status = api->Prepare(d.slumbot_pos, action, u, &ps);
      ++st.status_hist[status];
      if (!api->LastDraws().empty()) ++st.decisions_translated;
      if (status == SB_OK && api->LastTranslateStalled()) ++st.decisions_stalled;
      vector<std::pair<int, int>> hands = {{d.cards[0], d.cards[1]}};
      for (auto &h : ExtraHands(board, street, n_extra, rng)) hands.push_back(h);
      Card sorted[5];
      int board_street;
      SlumbotApi::SortBoard(board, kBoardCards[street], sorted, &board_street);
      string live_own;
      for (auto &h : hands) {
        string ms = MatchState(d.slumbot_pos, d.hand_no, action, h.first, h.second, board, street);
        int we_bet_to;
        BotAction ba = api->HandleStateChange(ms, &we_bet_to);
        string live = LiveIncr(ba, we_bet_to);
        if (&h == &hands[0]) live_own = live;
        if (status != SB_OK) {
          if (live.empty()) ++st.both_none;
          else {
            ++st.api_error_live_acted;
            Report(st, "api status " + std::to_string(status), ms, live, "-");
          }
          continue;
        }
        vector<OutAction> out;
        int r = api->QueryHand(ps, sorted, board_street, h.first, h.second, &out);
        string ai = r == 0 ? ApiIncr(out) : "";
        ++st.comparisons;
        if (r == 0 && ai.empty()) ++st.not_pure;
        if (live.empty()) {
          ++st.live_none;
          Report(st, "live none", ms, "-", ai);
        } else if (ai != live) {
          Report(st, "action", ms, live, ai);
        }
      }
      // The batched rollout call on Slumbot's own hand.
      if (status == SB_OK) {
        double r = 0.5;
        int type, bet, gst;
        sb_act_batch(api, 1, &d.slumbot_pos, action.c_str(), board, d.cards, &u, &r, &type, &bet,
                     &gst);
        string batch = gst != SB_OK ? "" : type == SB_FOLD ? "f" : type == SB_CHECK ? "k"
                       : type == SB_CALL ? "c" : "b" + std::to_string(bet);
        ++st.batch_checks;
        if (batch != live_own) {
          Report(st, "batch status " + std::to_string(gst), action, live_own, batch);
        }
      }
      // Cells of the history so far.
      vector<Cell> cells;
      int cstatus = api->Cells(d.slumbot_pos, action, &cells);
      ++st.cell_checks;
      bool ok = cstatus == SB_OK;
      if (ok) {
        ++st.cells_hist[(int)cells.size()];
        double mass = 0;
        int which = -1;
        for (size_t c = 0; c < cells.size(); ++c) {
          mass += cells[c].Mass();
          for (auto &iv : cells[c].intervals) {
            if (iv.first <= u && u < iv.second) which = (int)c;
          }
        }
        ok = std::fabs(mass - 1.0) < 1e-9 && which >= 0;
        if (ok && status == SB_OK) {
          ok = cells[which].feasible;
          PublicState ps2;
          ok = ok && api->Prepare(d.slumbot_pos, action, cells[which].Inside(), &ps2) == SB_OK &&
               ps2.path == ps.path && ps2.sob_node == ps.sob_node;
        }
      }
      if (!ok) {
        ++st.cell_failures;
        if (st.printed++ < 25) {
          printf("CELL FAILURE status %d pos %d hand %d action %s\n", cstatus, d.slumbot_pos,
                 d.hand_no, action.c_str());
        }
      }
    });
  }
  printf("hands %ld decisions %ld (translated %ld, stalled %ld) comparisons %ld\n", st.hands,
         st.decisions, st.decisions_translated, st.decisions_stalled, st.comparisons);
  printf("mismatches %ld not_pure %ld live_none %ld api_error_live_acted %ld both_none %ld\n",
         st.mismatches, st.not_pure, st.live_none, st.api_error_live_acted, st.both_none);
  printf("illegal: opponent %ld live %ld\n", st.opp_illegal, st.live_illegal);
  printf("cell checks %ld failures %ld, batch checks %ld\n", st.cell_checks, st.cell_failures,
         st.batch_checks);
  printf("prepare status:");
  for (auto &kv : st.status_hist) printf(" %d:%ld", kv.first, kv.second);
  printf("\ncells per history:");
  for (auto &kv : st.cells_hist) printf(" %d:%ld", kv.first, kv.second);
  printf("\n");
  return st.mismatches == 0 && st.cell_failures == 0 && st.not_pure == 0 ? 0 : 1;
}

int Bench(SlumbotApi *api, int n_states, uint64_t seed) {
  std::mt19937_64 rng(seed);
  struct Sample {
    int pos, street, hand_no;
    string action;
    int board[5];
  };
  vector<Sample> samples;
  Stats st;
  while ((int)samples.size() < n_states) {
    Deal d = RandomDeal(rng);
    PlayHand(api, d, rng, st, [&](const string &action, int street) {
      Sample s{d.slumbot_pos, street, d.hand_no, action, {}};
      std::copy(d.cards + 4, d.cards + 9, s.board);
      samples.push_back(s);
    });
  }
  samples.resize(n_states);
  using Clock = std::chrono::steady_clock;
  double t_query = 0, t_cells = 0, t_live = 0;
  long n_hands = 0, n_live = 0, n_ok = 0;
  vector<OutAction> out;
  std::map<int, std::pair<double, long>> by_street;
  for (const Sample &s : samples) {
    double u = sb_u_from_hand_no(s.hand_no);
    auto t0 = Clock::now();
    PublicState ps;
    int status = api->Prepare(s.pos, s.action, u, &ps);
    long hands_here = 0;
    if (status == SB_OK) {
      ++n_ok;
      Card sorted[5];
      int board_street;
      SlumbotApi::SortBoard(s.board, kBoardCards[s.street], sorted, &board_street);
      for (int c1 = 1; c1 < 52; ++c1) {
        for (int c2 = 0; c2 < c1; ++c2) {
          if (api->QueryHand(ps, sorted, board_street, c1, c2, &out) == 0) ++hands_here;
        }
      }
    }
    auto t1 = Clock::now();
    vector<Cell> cells;
    api->Cells(s.pos, s.action, &cells);
    auto t2 = Clock::now();
    // The live agent rebuilds everything per hand: time 20 hands of it.
    for (int k = 0; k < 20; ++k) {
      int c1 = rng() % 52, c2 = rng() % 52;
      bool ok = c1 != c2;
      for (int i = 0; i < kBoardCards[s.street]; ++i) ok = ok && s.board[i] != c1 && s.board[i] != c2;
      if (!ok) continue;
      int we_bet_to;
      api->HandleStateChange(MatchState(s.pos, s.hand_no, s.action, c1, c2, s.board, s.street),
                             &we_bet_to);
      ++n_live;
    }
    auto t3 = Clock::now();
    double dq = std::chrono::duration<double>(t1 - t0).count();
    t_query += dq;
    t_cells += std::chrono::duration<double>(t2 - t1).count();
    t_live += std::chrono::duration<double>(t3 - t2).count();
    n_hands += hands_here;
    by_street[s.street].first += dq;
    by_street[s.street].second += 1;
  }
  printf("states %d (prepare ok %ld), hands queried %ld\n", n_states, n_ok, n_hands);
  printf("query: %.3f ms/state, %.2f us/hand\n", 1e3 * t_query / n_states, 1e6 * t_query / n_hands);
  for (auto &kv : by_street) {
    printf("  street %d: %ld states, %.3f ms/state\n", kv.first, kv.second.second,
           1e3 * kv.second.first / kv.second.second);
  }
  printf("cells: %.1f us/state\n", 1e6 * t_cells / n_states);
  printf("live HandleStateChange: %.2f us/hand\n", 1e6 * t_live / n_live);
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <slumbot_dir> selfplay|bench|tree ...\n", argv[0]);
    return 2;
  }
  auto t0 = std::chrono::steady_clock::now();
  string err;
  SlumbotApi *api = sbapi::Create(argv[1], 16384, 22528, &err);
  if (api == nullptr) {
    fprintf(stderr, "load failed: %s\n", err.c_str());
    return 2;
  }
  printf("load %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  string mode = argv[2];
  int rc = 2;
  if (mode == "selfplay" && argc >= 5) {
    rc = SelfPlay(api, atoi(argv[3]), strtoull(argv[4], nullptr, 10), argc >= 6 ? atoi(argv[5]) : 8);
  } else if (mode == "bench" && argc >= 5) {
    rc = Bench(api, atoi(argv[3]), strtoull(argv[4], nullptr, 10));
  } else if (mode == "tree" && argc >= 5) {
    int n = api->ExportTree(string(argv[3]) == "sb", argv[4]);
    printf("tree %s: %d nodes\n", argv[3], n);
    rc = n > 0 ? 0 : 1;
  } else {
    fprintf(stderr, "bad arguments\n");
  }
  fflush(stdout);
  // Skip destructors: the slumbot globals are not meant to be torn down.
  _Exit(rc);
}
