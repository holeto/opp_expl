// C++ side of slumbot_api.h, shared by the library and the validator (validate.cpp), which needs
// the live agent (HandleStateChange) and the API on one object.
#ifndef SLUMBOT_API_IMPL_H_
#define SLUMBOT_API_IMPL_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "stateless_nl_agent.h"

class Node;

namespace sbapi {

// drand48: x <- (kA * x + kC) mod 2^48, seeded with x = (seed << 16) | 0x330E.
constexpr uint64_t kA = 0x5DEECE66DULL;
constexpr uint64_t kC = 0xBULL;
constexpr uint64_t kMask = (1ULL << 48) - 1;
constexpr double kTwo48 = 281474976710656.0;

// Seed-dependent part of the first draw: (kA * (s << 16)) mod 2^48. Linear in s modulo 2^32, so
// the draw for seed hand_no * 967 + off is ((U(hand_no) + D(off)) mod 2^48) / 2^48.
inline uint64_t SeedTerm(uint64_t s) { return (kA * ((s & 0xFFFFFFFFULL) << 16)) & kMask; }
inline uint64_t U48FromHandNo(int hand_no) {
  uint32_t s = static_cast<uint32_t>(hand_no) * 967U;
  return (SeedTerm(s) + kA * 0x330EULL + kC) & kMask;
}

// A union of disjoint intervals [lo, hi) in [0, 1), sorted.
using Intervals = std::vector<std::pair<double, double>>;

struct Cell {
  Intervals intervals;
  bool feasible;
  int num_draws;
  double Mass() const;
  double Inside() const;  // midpoint of the longest interval
};

struct Draw {
  uint64_t d48;      // D(off) of this translation
  double below_prob;
};

struct OutAction {
  int type;  // SB_FOLD, ...
  int bet;   // street bet-to for SB_BET
  double prob;
};

// The public part of a query, shared by all hands: the translated abstract path for one u.
struct PublicState {
  bool p1;
  std::string action_str;
  std::vector<Action> actions;
  std::vector<Node *> path;
  Node *sob_node;
  int last_bet_to;
  bool facing_bet;
  int previous_street_bet_to;
  int street;  // street of the next action (0 preflop .. 3 river)
};

class SlumbotApi : public StatelessNLAgent {
 public:
  using StatelessNLAgent::StatelessNLAgent;
  ~SlumbotApi() override;

  // Validates the action string, checks that `position` is to act and builds the path for u.
  int Prepare(int position, const std::string &action_str, double u, PublicState *ps);
  // Actions for one hand; board as SortBoard() returns it. Returns 0, 1 when the hand conflicts
  // with the board or itself (out left empty), or a negative status.
  int QueryHand(const PublicState &ps, const Card *board, int board_street, int c1, int c2,
                std::vector<OutAction> *out);
  int Cells(int position, const std::string &action_str, std::vector<Cell> *cells);
  int ExportTree(bool slumbot_is_sb, const std::string &out_path);

  // Sorts the board like ParseCardString (flop high to low) into board_out[5].
  static bool SortBoard(const int *board, int n_board, Card *board_out, int *board_street);

  // The draws of the last Prepare() (for tests).
  const std::vector<Draw> &LastDraws() const { return draws_; }
  // Whether the last walk ended at the opponent's node because Translate stalled (for tests).
  bool LastTranslateStalled() const { return last_translate_stalled_; }

 protected:
  double TranslationDraw(int hand_index, bool p1_choice, size_t path_size,
                         double below_prob) override;

 private:
  enum class Mode { kLive, kU, kScript };
  // Parses and checks the action string; `to_act` requires `position` to be the player to act.
  int ParseCommon(int position, const std::string &action_str, bool to_act, PublicState *ps);
  void ProcessActionsSafe(std::vector<Action> *actions, bool p1, bool cells_mode,
                          std::vector<Node *> *path, Node **sob_node, bool *terminate,
                          int *last_bet_to);
  void AllocBuckets();

  Mode mode_ = Mode::kLive;
  uint64_t u48_ = 0;
  std::vector<char> script_;  // forced outcomes of the first draws: 1 = below, 0 = above
  std::vector<Draw> draws_;
  bool last_translate_stalled_ = false;  // the last action's Translate left the path unchanged
  int ***buckets_ = nullptr;  // current_buckets[p1][street][bucketing]
};

// Loads the configuration globals and builds the agent (as query_strategy's main does).
SlumbotApi *Create(const std::string &slumbot_dir, int p1_it, int p2_it, std::string *err);

}  // namespace sbapi

#endif
