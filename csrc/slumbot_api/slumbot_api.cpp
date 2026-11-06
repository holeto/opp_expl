// Implementation of slumbot_api.h on top of StatelessNLAgent.
//
// The agent is used as the live bot uses it (HandleStateChange), with two changes that do not
// alter any answer the live bot gives:
//   - Translation draws come from TranslationDraw() (the hook in translation_hook.patch): either
//     frac(u + d_i) for a caller-supplied u, or forced outcomes while enumerating cells.
//   - The public path is built once per state (ProcessActionsSafe) and reused for every hand; the
//     live bot rebuilds it per query, but it depends on the actions and u only, never on the
//     cards. The loop stops at the first action that cannot be followed in the tree, where the
//     original would carry on and could index a missing successor. The live bot answers nothing
//     in that case anyway (terminate -> BA_NONE).
#include "slumbot_api.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "betting_abstraction.h"
#include "betting_abstraction_params.h"
#include "card_abstraction.h"
#include "card_abstraction_params.h"
#include "files.h"
#include "game.h"
#include "game_params.h"
#include "nonterminal_ids.h"
#include "params.h"
#include "poker_action.h"
#include "protocol.h"
#include "runtime_config.h"
#include "runtime_params.h"
#include "slumbot_api_impl.h"
#include "slumbot_rand.h"
#include "training_config.h"
#include "training_params.h"
#include "tree.h"

using std::string;
using std::vector;

namespace sbapi {

// ── Interval sets on [0, 1) ──────────────────────────────────────────────────

static double Measure(const Intervals &s) {
  double m = 0;
  for (const auto &iv : s) m += iv.second - iv.first;
  return m;
}

static Intervals Intersect(const Intervals &a, const Intervals &b) {
  Intervals out;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    double lo = std::max(a[i].first, b[j].first);
    double hi = std::min(a[i].second, b[j].second);
    if (lo < hi) out.push_back({lo, hi});
    if (a[i].second < b[j].second) ++i;
    else ++j;
  }
  return out;
}

static Intervals Complement(const Intervals &s) {
  Intervals out;
  double prev = 0;
  for (const auto &iv : s) {
    if (iv.first > prev) out.push_back({prev, iv.first});
    prev = iv.second;
  }
  if (prev < 1.0) out.push_back({prev, 1.0});
  return out;
}

// {u : frac(u + d) < p}: the arc [frac(-d), frac(-d) + p) of the circle, as at most two intervals.
static Intervals BelowArc(double d, double p) {
  if (p <= 0) return {};
  if (p >= 1) return {{0.0, 1.0}};
  double start = 1.0 - d;
  if (start >= 1.0) start -= 1.0;
  double end = start + p;
  if (end <= 1.0) return {{start, end}};
  return {{0.0, end - 1.0}, {start, 1.0}};
}

double Cell::Mass() const { return Measure(intervals); }

double Cell::Inside() const {
  size_t best = 0;
  for (size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].second - intervals[i].first >
        intervals[best].second - intervals[best].first) {
      best = i;
    }
  }
  return 0.5 * (intervals[best].first + intervals[best].second);
}

// ── Construction ──────────────────────────────────────────────────────────────

static void AssignIds(Tree *tree) {
  unsigned int ***num_nonterminals;
  AssignNonterminalIDs(tree, &num_nonterminals);
  int max_street = Game::MaxStreet();
  for (int p1 = 0; p1 <= 1; ++p1) {
    for (int st = 0; st <= max_street; ++st) delete[] num_nonterminals[p1][st];
    delete[] num_nonterminals[p1];
  }
  delete[] num_nonterminals;
}

static bool IsDir(const string &path) {
  struct stat sb;
  return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

SlumbotApi *Create(const string &dir, int p1_it, int p2_it, string *err) {
  // Game, BettingAbstraction, TrainingConfig and Files are process-wide singletons.
  static bool created = false;
  if (created) {
    *err = "Slumbot can be loaded only once per process";
    return nullptr;
  }
  const char *names[] = {"holdem_params",      "p1_card_params",   "p2_card_params",
                         "nlt5_params",        "qsqtcfr_psp_params", "runtime_ui_params"};
  vector<string> paths;
  for (const char *name : names) {
    paths.push_back(dir + "/" + name);
    // The slumbot reader exits on a missing file; check here so that a wrong path is an error.
    if (access(paths.back().c_str(), R_OK) != 0) {
      *err = "cannot read " + paths.back();
      return nullptr;
    }
  }
  setenv("SLUMBOT_DATA", (dir + "/data").c_str(), 0);  // keeps a caller's $SLUMBOT_DATA
  string data = getenv("SLUMBOT_DATA");
  if (!IsDir(data + "/static") || !IsDir(data + "/training")) {
    *err = "no static/ and training/ under " + data;
    return nullptr;
  }

  Files::Init();
  Params *game_params = CreateGameParams();
  game_params->ReadFromFile(paths[0].c_str());
  Game::Initialize(game_params);
  Params *p1_cap = CreateCardAbstractionParams();
  p1_cap->ReadFromFile(paths[1].c_str());
  CardAbstractionInstance *p1_cai = new CardAbstractionInstance(p1_cap);
  Params *p2_cap = CreateCardAbstractionParams();
  p2_cap->ReadFromFile(paths[2].c_str());
  CardAbstractionInstance *p2_cai = new CardAbstractionInstance(p2_cap);
  Params *bap = CreateBettingAbstractionParams();
  bap->ReadFromFile(paths[3].c_str());
  BettingAbstraction::Initialize(bap);
  Params *tp = CreateTrainingParams();
  tp->ReadFromFile(paths[4].c_str());
  TrainingConfig::Initialize(tp);
  Params *rp = CreateRuntimeParams();
  rp->ReadFromFile(paths[5].c_str());
  RuntimeConfigInstance *p1_rci = new RuntimeConfigInstance(rp);
  RuntimeConfigInstance *p2_rci = new RuntimeConfigInstance(rp);
  p1_rci->SetIteration(p1_it);
  p2_rci->SetIteration(p2_it);
  InitRandFixed();

  if (!BettingAbstraction::Asymmetric()) {
    *err = "expected an asymmetric betting abstraction";
    return nullptr;
  }
  Tree *p1_tree = Tree::BuildAsymmetricTree(true);
  Tree *p2_tree = Tree::BuildAsymmetricTree(false);
  AssignIds(p1_tree);
  AssignIds(p2_tree);

  // Blinds 50/100 and stack 20000: slumbot.com units, as query_strategy uses them.
  SlumbotApi *api = new SlumbotApi(p1_cai, p2_cai, BettingAbstraction::Instance(),
                                   TrainingConfig::Instance(), p1_rci, p2_rci, p1_tree, p2_tree,
                                   /*debug=*/false, /*exit_on_error=*/false, 50, 20000);
  created = true;
  return api;
}

// ── Translation draws ────────────────────────────────────────────────────────

double SlumbotApi::TranslationDraw(int hand_index, bool p1_choice, size_t path_size,
                                   double below_prob) {
  if (mode_ == Mode::kLive) {
    return StatelessNLAgent::TranslationDraw(hand_index, p1_choice, path_size, below_prob);
  }
  // Same integer expression as the seed, minus the hand-number term.
  uint64_t off = static_cast<uint64_t>((p1_choice ? 23 : 0) + path_size * 7);
  uint64_t d48 = SeedTerm(off);
  size_t k = draws_.size();
  draws_.push_back({d48, below_prob});
  if (mode_ == Mode::kU) return static_cast<double>((u48_ + d48) & kMask) / kTwo48;
  // kScript: r = 0 picks the below size (below_prob > 0 on any branch we force), r = 1 the above.
  if (k < script_.size()) return script_[k] ? 0.0 : 1.0;
  return 0.0;  // past the script: recorded only, the enumeration branches on it next
}

// ── Public state ──────────────────────────────────────────────────────────────

int SlumbotApi::ParseCommon(int position, const string &action_str, bool to_act,
                            PublicState *ps) {
  if (position != 0 && position != 1) return SB_ERR_ARGS;
  int pos, st, street_last_bet_to, total_last_bet_to, last_bet_size, last_bettor;
  ActionError error;
  if (!ActionState(action_str, pos, st, small_blind_, 2 * small_blind_, stack_size_,
                   street_last_bet_to, total_last_bet_to, last_bet_size, last_bettor, error) ||
      error != ActionError::NO_ERROR) {
    return SB_ERR_ACTION;
  }
  if (to_act && pos != position) return SB_ERR_NOT_TO_ACT;
  ps->p1 = position == 1;
  ps->action_str = action_str;
  ps->street = st;
  ps->facing_bet = last_bettor == 1 - pos;  // FacingBet()
  ps->previous_street_bet_to = total_last_bet_to - street_last_bet_to;
  if (!ParseActions(action_str, 2 * small_blind_, &ps->actions)) return SB_ERR_ACTION;
  if (to_act) {
    // The live bot's own checks, in its order.
    if (AreWeAllIn(&ps->actions)) return SB_ERR_NOT_TO_ACT;
    bool p1_to_act, p2_to_act;
    WhoseAction(&ps->actions, &p1_to_act, &p2_to_act);
    if (!p1_to_act && !p2_to_act) return SB_ERR_NOT_TO_ACT;
    if (ps->p1 != p1_to_act) return SB_ERR_NOT_TO_ACT;
  }
  return SB_OK;
}

// StatelessNLAgent::ProcessActions, stopping at the first action that cannot be followed.
// cells_mode: actions after a forced all-in end the walk without marking it off-tree (the live
// bot only ever sees the opponent's last action there, and answers nothing).
void SlumbotApi::ProcessActionsSafe(vector<Action> *actions, bool p1, bool cells_mode,
                                    vector<Node *> *path, Node **sob_node, bool *terminate,
                                    int *last_bet_to) {
  *sob_node = nullptr;
  *last_bet_to = 2 * small_blind_;
  path->clear();
  path->push_back(p1 ? p1_tree_->Root() : p2_tree_->Root());
  *terminate = false;
  bool forced_all_in = false;
  int opp_bet_amount = 0;
  bool p1_to_act = true;
  bool street_initial = true;
  last_translate_stalled_ = false;
  for (const Action &a : *actions) {
    Node *current = path->back();
    last_translate_stalled_ = false;
    if (forced_all_in) {
      if (!cells_mode) *terminate = true;
      return;
    }
    bool check_or_call = a.action_type == ActionType::CALL || a.action_type == ActionType::CHECK;
    bool ours = p1_to_act == p1;
    // A terminal node is legal only for our action right after a small bet that Translate
    // mapped to a check/call (sob_node), e.g. a river min-raise mapped to a showdown call:
    // Interpret() then skips our check/call/fold, or replaces the node with sob_node when we
    // raise. Anything else would index a missing successor.
    if (current->Terminal() && !(ours && *sob_node != nullptr)) {
      *terminate = true;
      return;
    }
    if (ours) {
      Interpret(a.action_type, a.total_bet_to, path, *sob_node, *last_bet_to, opp_bet_amount,
                &forced_all_in, terminate);
      *sob_node = nullptr;
      opp_bet_amount = 0;
      if (*terminate) return;
    } else {
      int s = a.action_type == ActionType::FOLD ? current->FoldSuccIndex()
              : check_or_call                   ? current->CallSuccIndex()
                                                : 0;
      if (s < 0) {
        *terminate = true;
        return;
      }
      size_t before = path->size();
      // The hand index is unused: draws come from TranslationDraw().
      Translate(a.action_type, a.total_bet_to, path, sob_node, 2 * *last_bet_to, 0);
      // Translate gives up ("No smallest bet succ?!?") on a raise at a node with no bet succ,
      // leaving the path at the opponent's node. The live agent carries on from there and plays
      // that node's strategy, so we do the same; Interpret() terminates safely on what it
      // cannot follow.
      last_translate_stalled_ = path->size() == before;
      opp_bet_amount = a.action_type == ActionType::BET ? a.total_bet_to - *last_bet_to : 0;
    }
    if (a.action_type == ActionType::BET) *last_bet_to = a.total_bet_to;
    if (check_or_call && !street_initial) {
      p1_to_act = false;
      street_initial = true;
    } else {
      p1_to_act = !p1_to_act;
      street_initial = false;
    }
  }
}

int SlumbotApi::Prepare(int position, const string &action_str, double u, PublicState *ps) {
  int status = ParseCommon(position, action_str, true, ps);
  if (status != SB_OK) return status;
  if (!(u >= 0.0 && u < 1.0)) return SB_ERR_ARGS;
  mode_ = Mode::kU;
  u48_ = static_cast<uint64_t>(u * kTwo48) & kMask;
  draws_.clear();
  bool terminate;
  ProcessActionsSafe(&ps->actions, ps->p1, false, &ps->path, &ps->sob_node, &terminate,
                     &ps->last_bet_to);
  mode_ = Mode::kLive;
  if (terminate) return SB_ERR_OFF_TREE;
  // Without a pending small-bet decision the path must end at one of our own decisions, or at
  // the opponent's node where Translate stalled on their last action (see ProcessActionsSafe).
  Node *node = ps->path.back();
  if (ps->sob_node == nullptr &&
      (node->Terminal() || (node->P1Choice() != ps->p1 && !last_translate_stalled_))) {
    return SB_ERR_OFF_TREE;
  }
  return SB_OK;
}

void SlumbotApi::AllocBuckets() {
  int max_street = Game::MaxStreet();
  buckets_ = new int **[2];
  for (int p1 = 0; p1 <= 1; ++p1) {
    buckets_[p1] = new int *[max_street + 1];
    for (int st = 0; st <= max_street; ++st) buckets_[p1][st] = new int[num_bucketings_[st]];
  }
}

SlumbotApi::~SlumbotApi() {
  if (buckets_ == nullptr) return;
  int max_street = Game::MaxStreet();
  for (int p1 = 0; p1 <= 1; ++p1) {
    for (int st = 0; st <= max_street; ++st) delete[] buckets_[p1][st];
    delete[] buckets_[p1];
  }
  delete[] buckets_;
}

static const int kBoardCards[4] = {0, 3, 4, 5};

bool SlumbotApi::SortBoard(const int *board, int n_board, Card *out, int *board_street) {
  int st = 0;
  while (st < 4 && kBoardCards[st] != n_board) ++st;
  if (st == 4) return false;
  for (int i = 0; i < 5; ++i) out[i] = 0;
  for (int i = 0; i < n_board; ++i) {
    if (board[i] < 0 || board[i] > 51) return false;
    for (int j = 0; j < i; ++j) {
      if (board[i] == board[j]) return false;
    }
    out[i] = static_cast<Card>(board[i]);
  }
  // ParseCardString orders the flop high to low; the bucket lookup depends on it.
  if (n_board >= 3) std::sort(out, out + 3, [](Card a, Card b) { return a > b; });
  *board_street = st;
  return true;
}

int SlumbotApi::QueryHand(const PublicState &ps, const Card *board, int board_street, int c1,
                          int c2, vector<OutAction> *out) {
  out->clear();
  if (c1 < 0 || c1 > 51 || c2 < 0 || c2 > 51 || c1 == c2) return 1;
  for (int i = 0; i < kBoardCards[board_street]; ++i) {
    if (board[i] == c1 || board[i] == c2) return 1;
  }
  if (buckets_ == nullptr) AllocBuckets();
  Card cards[5];
  std::copy(board, board + 5, cards);
  UpdateCards(board_street, static_cast<Card>(std::max(c1, c2)),
              static_cast<Card>(std::min(c1, c2)), cards, buckets_);

  // As HandleStateChange from GetActionProbs on, with the distribution instead of a sample.
  vector<Node *> path = ps.path;  // GetActionProbs may replace the last node (HandleRaise)
  int opp_bet_amount = 0;
  bool force_all_in = false, force_call = false;
  double *probs = GetActionProbs(ps.actions, &path, ps.sob_node, 0, buckets_, ps.p1,
                                 &opp_bet_amount, &force_all_in, &force_call);
  Node *node = path.back();
  int call_type = ps.facing_bet ? SB_CALL : SB_CHECK;
  if (force_call) {
    out->push_back({call_type, 0, 1.0});
    delete[] probs;
    return 0;
  }
  if (probs == nullptr) return SB_ERR_INTERNAL;
  HardCodedResponse(node, ps.p1, buckets_, probs);
  int num_succs = node->NumSuccs();
  int csi = node->CallSuccIndex();
  int fsi = node->FoldSuccIndex();
  for (int s = 0; s < num_succs; ++s) {
    if (s == csi) {
      if (force_all_in) {
        out->push_back({SB_BET, stack_size_ - ps.previous_street_bet_to, probs[s]});
      } else {
        out->push_back({call_type, 0, probs[s]});
      }
    } else if (s == fsi) {
      out->push_back({SB_FOLD, 0, probs[s]});
    } else {
      int we_bet_to = GetWeBetTo(node->IthSucc(s), ps.last_bet_to, opp_bet_amount);
      if (we_bet_to == 0) {
        out->push_back({call_type, 0, probs[s]});
      } else {
        out->push_back({SB_BET, we_bet_to - ps.previous_street_bet_to, probs[s]});
      }
    }
  }
  delete[] probs;
  return 0;
}

// ── Cells ──────────────────────────────────────────────────────────────────────

int SlumbotApi::Cells(int position, const string &action_str, vector<Cell> *cells) {
  PublicState ps;
  int status = ParseCommon(position, action_str, false, &ps);
  if (status != SB_OK) return status;
  cells->clear();
  // Depth-first over draw outcomes. A pending entry fixes the outcomes of the first draws and
  // the set of u that produces them; replaying the actions with those outcomes forced shows the
  // next draw (if any), whose arc splits the set in two. The draw sequence before draw k is the
  // same for every u in the set, so the split is exact.
  struct Pending {
    vector<char> script;
    Intervals set;
  };
  vector<Pending> stack;
  stack.push_back({{}, {{0.0, 1.0}}});
  vector<vector<char>> scripts;
  while (!stack.empty()) {
    Pending p = std::move(stack.back());
    stack.pop_back();
    mode_ = Mode::kScript;
    script_ = p.script;
    draws_.clear();
    vector<Action> actions = ps.actions;
    vector<Node *> path;
    Node *sob_node;
    bool terminate;
    int last_bet_to;
    ProcessActionsSafe(&actions, ps.p1, true, &path, &sob_node, &terminate, &last_bet_to);
    mode_ = Mode::kLive;
    size_t k = p.script.size();
    if (draws_.size() > k) {
      double d = static_cast<double>(draws_[k].d48) / kTwo48;
      Intervals below_arc = BelowArc(d, draws_[k].below_prob);
      Intervals above = Intersect(p.set, Complement(below_arc));
      Intervals below = Intersect(p.set, below_arc);
      if (Measure(above) > 0) {
        stack.push_back({p.script, above});
        stack.back().script.push_back(0);
      }
      if (Measure(below) > 0) {
        stack.push_back({p.script, below});
        stack.back().script.push_back(1);
      }
    } else {
      cells->push_back({p.set, !terminate, static_cast<int>(k)});
      scripts.push_back(p.script);
    }
  }
  // The point handed out for each cell must reproduce its outcomes through the u path.
  for (size_t c = 0; c < cells->size(); ++c) {
    u48_ = static_cast<uint64_t>((*cells)[c].Inside() * kTwo48) & kMask;
    mode_ = Mode::kU;
    draws_.clear();
    vector<Action> actions = ps.actions;
    vector<Node *> path;
    Node *sob_node;
    bool terminate;
    int last_bet_to;
    ProcessActionsSafe(&actions, ps.p1, true, &path, &sob_node, &terminate, &last_bet_to);
    mode_ = Mode::kLive;
    if (draws_.size() != scripts[c].size()) return SB_ERR_INTERNAL;
    for (size_t i = 0; i < draws_.size(); ++i) {
      double r = static_cast<double>((u48_ + draws_[i].d48) & kMask) / kTwo48;
      if ((r < draws_[i].below_prob) != (scripts[c][i] == 1)) return SB_ERR_INTERNAL;
    }
  }
  return SB_OK;
}

// ── Tree export ────────────────────────────────────────────────────────────────

int SlumbotApi::ExportTree(bool slumbot_is_sb, const string &out_path) {
  FILE *f = fopen(out_path.c_str(), "w");
  if (f == nullptr) return SB_ERR_ARGS;
  fprintf(f, "# id parent succ_index street p1_choice n_succs pot_size label\n");
  int next_id = 0;
  struct Item {
    Node *node;
    int parent;
    int succ;
    string label;
  };
  vector<Item> stack;
  Tree *tree = slumbot_is_sb ? p1_tree_ : p2_tree_;
  stack.push_back({tree->Root(), -1, -1, "root"});
  while (!stack.empty()) {
    Item it = std::move(stack.back());
    stack.pop_back();
    int id = next_id++;
    Node *n = it.node;
    fprintf(f, "%d %d %d %u %d %u %d %s\n", id, it.parent, it.succ, n->Street(),
            static_cast<int>(n->P1Choice()), n->NumSuccs(), n->PotSize(), it.label.c_str());
    int num_succs = n->NumSuccs();
    int csi = n->CallSuccIndex();
    int fsi = n->FoldSuccIndex();
    for (int s = num_succs - 1; s >= 0; --s) {  // reversed so that succ 0 is written first
      Node *succ = n->IthSucc(s);
      string label;
      if (s == csi) {
        label = "c";
      } else if (s == fsi) {
        label = "f";
      } else {
        Node *call = succ->IthSucc(succ->CallSuccIndex());
        label = "b" + std::to_string((call->PotSize() / 2) * small_blind_);
      }
      stack.push_back({succ, id, s, label});
    }
  }
  fclose(f);
  return next_id;
}

}  // namespace sbapi

// ── C interface ──────────────────────────────────────────────────────────────

using sbapi::SlumbotApi;

static void SetErr(char *err, int err_len, const string &msg) {
  if (err != nullptr && err_len > 0) {
    std::snprintf(err, err_len, "%s", msg.c_str());
  }
}

extern "C" {

void *sb_create(const char *slumbot_dir, int p1_it, int p2_it, char *err, int err_len) {
  try {
    string msg;
    SlumbotApi *api = sbapi::Create(slumbot_dir, p1_it, p2_it, &msg);
    if (api == nullptr) SetErr(err, err_len, msg);
    return api;
  } catch (const std::exception &e) {
    SetErr(err, err_len, e.what());
    return nullptr;
  }
}

void sb_destroy(void *handle) { delete static_cast<SlumbotApi *>(handle); }

double sb_u_from_hand_no(int hand_no) {
  return static_cast<double>(sbapi::U48FromHandNo(hand_no)) / sbapi::kTwo48;
}

int sb_query(void *handle, int position, const char *actions, const int *board, int n_board,
             const int *hands, int n_hands, double u, int max_out, int *out_n, int *out_type,
             int *out_bet, double *out_prob) {
  if (handle == nullptr || actions == nullptr || n_hands < 0 || max_out <= 0) return SB_ERR_ARGS;
  try {
    SlumbotApi *api = static_cast<SlumbotApi *>(handle);
    sbapi::PublicState ps;
    int status = api->Prepare(position, actions, u, &ps);
    if (status != SB_OK) return status;
    Card sorted[5];
    int board_street;
    if (!SlumbotApi::SortBoard(board, n_board, sorted, &board_street) ||
        board_street != ps.street) {
      return SB_ERR_ARGS;
    }
    vector<sbapi::OutAction> out;
    for (int h = 0; h < n_hands; ++h) {
      int r = api->QueryHand(ps, sorted, board_street, hands[2 * h], hands[2 * h + 1], &out);
      if (r < 0) return r;
      if (static_cast<int>(out.size()) > max_out) return SB_ERR_CAPACITY;
      out_n[h] = static_cast<int>(out.size());
      for (size_t i = 0; i < out.size(); ++i) {
        out_type[h * max_out + i] = out[i].type;
        out_bet[h * max_out + i] = out[i].bet;
        out_prob[h * max_out + i] = out[i].prob;
      }
    }
    return SB_OK;
  } catch (const std::exception &) {
    return SB_ERR_INTERNAL;
  }
}

int sb_act_batch(void *handle, int n, const int *position, const char *actions,
                 const int *boards, const int *hands, const double *u, const double *r,
                 int *out_type, int *out_bet, int *status) {
  if (handle == nullptr || actions == nullptr || n < 0) return SB_ERR_ARGS;
  SlumbotApi *api = static_cast<SlumbotApi *>(handle);
  const char *a = actions;
  vector<sbapi::OutAction> out;
  for (int g = 0; g < n; ++g) {
    string action_str(a);
    a += action_str.size() + 1;
    out_type[g] = out_bet[g] = 0;
    try {
      sbapi::PublicState ps;
      int st = api->Prepare(position[g], action_str, u[g], &ps);
      Card sorted[5];
      int board_street;
      static const int kCards[4] = {0, 3, 4, 5};
      if (st == SB_OK && (ps.street < 0 || ps.street > 3 ||
                          !SlumbotApi::SortBoard(boards + 5 * g, kCards[ps.street], sorted,
                                                 &board_street))) {
        st = SB_ERR_ARGS;
      }
      if (st == SB_OK) {
        st = api->QueryHand(ps, sorted, board_street, hands[2 * g], hands[2 * g + 1], &out);
        // A hand that conflicts with the board (1) is the caller's error here.
        if (st == 1 || out.empty()) st = SB_ERR_ARGS;
      }
      if (st == SB_OK) {
        // The live agent's sampling: the first successor whose cumulative
        // probability exceeds r, else the last.
        size_t s = 0;
        double cum = 0.0;
        for (; s + 1 < out.size(); ++s) {
          cum += out[s].prob;
          if (r[g] < cum) break;
        }
        out_type[g] = out[s].type;
        out_bet[g] = out[s].bet;
      }
      status[g] = st;
    } catch (const std::exception &) {
      status[g] = SB_ERR_INTERNAL;
    }
  }
  return SB_OK;
}

int sb_cells(void *handle, int position, const char *actions, int max_cells, int max_intervals,
             int *n_cells, int *cell_feasible, double *cell_mass, double *cell_u, int *cell_draws,
             int *n_intervals, int *interval_cell, double *interval_lo, double *interval_hi) {
  if (handle == nullptr || actions == nullptr) return SB_ERR_ARGS;
  try {
    SlumbotApi *api = static_cast<SlumbotApi *>(handle);
    vector<sbapi::Cell> cells;
    int status = api->Cells(position, actions, &cells);
    if (status != SB_OK) return status;
    struct Iv {
      double lo, hi;
      int cell;
    };
    vector<Iv> ivs;
    for (size_t c = 0; c < cells.size(); ++c) {
      for (const auto &iv : cells[c].intervals) ivs.push_back({iv.first, iv.second, (int)c});
    }
    if ((int)cells.size() > max_cells || (int)ivs.size() > max_intervals) return SB_ERR_CAPACITY;
    std::sort(ivs.begin(), ivs.end(), [](const Iv &a, const Iv &b) { return a.lo < b.lo; });
    *n_cells = static_cast<int>(cells.size());
    for (size_t c = 0; c < cells.size(); ++c) {
      cell_feasible[c] = cells[c].feasible ? 1 : 0;
      cell_mass[c] = cells[c].Mass();
      cell_u[c] = cells[c].Inside();
      cell_draws[c] = cells[c].num_draws;
    }
    *n_intervals = static_cast<int>(ivs.size());
    for (size_t i = 0; i < ivs.size(); ++i) {
      interval_cell[i] = ivs[i].cell;
      interval_lo[i] = ivs[i].lo;
      interval_hi[i] = ivs[i].hi;
    }
    return SB_OK;
  } catch (const std::exception &) {
    return SB_ERR_INTERNAL;
  }
}

int sb_export_tree(void *handle, int slumbot_is_sb, const char *out_path) {
  if (handle == nullptr || out_path == nullptr) return SB_ERR_ARGS;
  try {
    return static_cast<SlumbotApi *>(handle)->ExportTree(slumbot_is_sb != 0, out_path);
  } catch (const std::exception &) {
    return SB_ERR_INTERNAL;
  }
}

}  // extern "C"
