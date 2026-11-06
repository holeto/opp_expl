/* C interface to Slumbot's strategy (Eric Jackson's HUNL agent behind slumbot.com), built on the
 * slumbot_query package. Loaded from Python with ctypes.
 *
 * Units and conventions are slumbot.com's, so that states and actions round-trip with the live
 * API unchanged:
 *   - Blinds 50/100, stacks 20000. Position 0 is the big blind, 1 the small blind.
 *   - Actions are the slumbot.com action string ("b200c/kb300", ...): k check, c call, f fold,
 *     b<N> bet or raise to a street total of N chips, '/' between streets.
 *   - Cards are rank * 4 + suit, ranks 2..A = 0..12, suits c, d, h, s = 0..3 (the env's encoding).
 *
 * Determinism. The stored strategy is pure: one action per (state, hand). The only randomness is
 * translation of an opponent bet that falls between two abstract sizes: Slumbot takes the smaller
 * one iff r < below_prob. The live agent draws r from drand48 seeded with
 * hand_no * 967 + 23 * [opponent is SB] + 7 * path_len, and the first drand48 draw is affine in the
 * seed modulo 1. So every draw of a hand is frac(u + d_i), with a single u in [0, 1) per hand and
 * offsets d_i that depend only on the path. These functions take u directly: Slumbot's policy
 * is a mixture over u ~ U[0, 1), and sb_cells() partitions [0, 1) into the cells on which the
 * translated path is constant. sb_u_from_hand_no() gives the u of a live hand number (exact).
 *
 * Not thread-safe: the agent keeps one file handle per strategy file and seeks it per lookup.
 * One handle per process: the slumbot code keeps its configuration in globals.
 */
#ifndef SLUMBOT_API_H_
#define SLUMBOT_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. Per call; per-hand problems are reported through out_n == 0 instead. */
#define SB_OK 0
#define SB_ERR_ARGS -1       /* bad handle, position, board size, capacity */
#define SB_ERR_ACTION -2     /* action string illegal under slumbot.com rules */
#define SB_ERR_NOT_TO_ACT -3 /* hand over, or the other player is to act */
#define SB_ERR_OFF_TREE -4   /* a past Slumbot action cannot be matched in its tree */
#define SB_ERR_CAPACITY -5   /* max_out / max_cells / max_intervals too small */
#define SB_ERR_INTERNAL -6

/* Output action types. */
#define SB_FOLD 0
#define SB_CHECK 1
#define SB_CALL 2
#define SB_BET 3

/* Loads the agent: parameter files from slumbot_dir, strategy data from $SLUMBOT_DATA, which
 * defaults to <slumbot_dir>/data. p1_it / p2_it are the CFR iterations of the SB / BB
 * strategies (16384 / 22528 for the shipped data). Returns NULL and fills err on failure. A
 * missing or corrupt data file can still terminate the process: the slumbot reader exits on I/O
 * errors. */
void *sb_create(const char *slumbot_dir, int p1_it, int p2_it, char *err, int err_len);
void sb_destroy(void *handle);

/* The u that the live agent uses in hand hand_no (exact: u * 2^48 is an integer). */
double sb_u_from_hand_no(int hand_no);

/* Slumbot's action distribution at one public state, for n_hands of its own hands.
 *   position: Slumbot's seat (0 BB, 1 SB); it must be the player to act.
 *   board: n_board cards (0, 3, 4 or 5, matching the street of `actions`).
 *   hands: n_hands x 2 cards.
 *   u: translation type in [0, 1).
 * For hand h, out_n[h] actions are written to row h of the n_hands x max_out arrays: type,
 * street bet-to (the N of b<N>; 0 unless SB_BET) and probability. The rows list every successor of
 * Slumbot's abstract node, zeros included, so they also expose its action set; distinct
 * successors can map to the same chips (two bets capped at all-in). out_n[h] = 0 marks a hand that
 * conflicts with the board or itself: Slumbot itself answers for such hands. The actions follow the
 * live agent (HandleStateChange), which differs from QueryStrategy in one case: after translating
 * a bet up to all-in before the river, Slumbot shoves instead of calling. */
int sb_query(void *handle, int position, const char *actions, const int *board, int n_board,
             const int *hands, int n_hands, double u, int max_out, int *out_n, int *out_type,
             int *out_bet, double *out_prob);

/* Slumbot's action in each of n independent games (for rollouts). Game g has its own position,
 * action string, board (boards + 5 * g: the full board; the first 0/3/4/5 cards are used per the
 * street of its string), hand, translation type u[g] and uniform r[g], with which the action is
 * drawn from its distribution as the live agent draws it (the strategy is pure, so r matters only
 * in degenerate cases). `actions` holds the n strings back to back, each NUL-terminated. Writes
 * out_type[g] / out_bet[g] (as in sb_query) and a per-game status[g]; a hand that conflicts with
 * the board is SB_ERR_ARGS. Returns SB_ERR_ARGS for bad arguments, else SB_OK. */
int sb_act_batch(void *handle, int n, const int *position, const char *actions,
                 const int *boards, const int *hands, const double *u, const double *r,
                 int *out_type, int *out_bet, int *status);

/* The translation cells of a public state: the partition of u in [0, 1) by the outcomes of all
 * opponent-bet translations along `actions` (opponent = not `position`). Each cell is a union of
 * intervals [lo, hi); the path, and so Slumbot's strategy at this state, is constant on it.
 *   cell_feasible[c] = 0: on that cell some past Slumbot action is not in its tree, so the cell
 *     has likelihood 0 for this history.
 *   cell_mass[c]: total length; the masses sum to 1.
 *   cell_u[c]: a point inside the cell (midpoint of its longest interval), to pass to sb_query.
 *   cell_draws[c]: number of translation draws on the cell's path.
 *   interval_cell / interval_lo / interval_hi: the n_intervals intervals, sorted by lo.
 * `position` is Slumbot's seat; it need not be the player to act, so cells of a terminal or
 * opponent-to-act history are available too. */
int sb_cells(void *handle, int position, const char *actions, int max_cells, int max_intervals,
             int *n_cells, int *cell_feasible, double *cell_mass, double *cell_u, int *cell_draws,
             int *n_intervals, int *interval_cell, double *interval_lo, double *interval_hi);

/* Writes Slumbot's betting tree for its seat (slumbot_is_sb = 1: the P1 tree) to out_path, one
 * node per line in depth-first order:
 *   id parent succ_index street p1_choice n_succs pot_size label
 * p1_choice = 1 if the SB acts; pot_size is the tree's own field (twice the matched bet, in
 * units of the abstract small blind, so chips = pot_size / 2 * 50 per player); label is
 * root, f, c (check or call) or b<total bet-to in chips>. Returns the number of nodes or an error. */
int sb_export_tree(void *handle, int slumbot_is_sb, const char *out_path);

#ifdef __cplusplus
}
#endif

#endif
