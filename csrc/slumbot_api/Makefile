# Builds libslumbot_api.so (for ctypes) and the validator against a slumbot_query checkout whose
# objects are built with translation_hook.patch applied:
#   cd $SLUMBOT_DIR && git apply <this dir>/translation_hook.patch && make binaries
#   make SLUMBOT_DIR=/path/to/slumbot_query
# STATIC_RT links libstdc++ and libgcc into the library, so that it loads into a Python process
# whatever libstdc++ that process already has; set STATIC_RT= to link them dynamically.

SLUMBOT_DIR ?= $(error set SLUMBOT_DIR to the slumbot_query directory)
BUILD ?= build
CXX ?= g++
CXXFLAGS = -std=c++17 -Wall -O3 -march=x86-64-v2 -fPIC -I$(SLUMBOT_DIR)/src
STATIC_RT ?= -static-libstdc++ -static-libgcc

# Every slumbot object except the programs' mains (the OBJS list of its Makefile).
SB_OBJS = $(addprefix $(SLUMBOT_DIR)/obj/, split.o match_state.o cards.o io.o params.o game.o \
	files.o poker_action.o hand_tree.o canonical_cards.o boards.o blob.o array.o \
	card_abstraction.o partitions.o buckets.o hands.o betting_abstraction.o pool.o tree.o \
	nonterminal_ids.o training_config.o slumbot_rand.o pprobs.o probs.o runtime_config.o \
	nearest_neighbors.o protocol.o agent_buckets.o translation.o alternative.o respond.o \
	hard_coded.o stateless_nl_agent.o betting_abstraction_params.o card_abstraction_params.o \
	game_params.o runtime_params.o training_params.o)

HEADS = slumbot_api.h slumbot_api_impl.h

all: $(BUILD)/libslumbot_api.so $(BUILD)/validate

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/slumbot_api.o: slumbot_api.cpp $(HEADS) $(SLUMBOT_DIR)/src/stateless_nl_agent.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/validate.o: validate.cpp $(HEADS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# --exclude-libs keeps the static runtime's symbols out of the library's exports.
$(BUILD)/libslumbot_api.so: $(BUILD)/slumbot_api.o $(SB_OBJS)
	$(CXX) -shared -o $@ $^ $(STATIC_RT) -Wl,--no-undefined -Wl,--exclude-libs,ALL -pthread

$(BUILD)/validate: $(BUILD)/validate.o $(BUILD)/slumbot_api.o $(SB_OBJS)
	$(CXX) -o $@ $^ $(STATIC_RT) -pthread

clean:
	rm -rf $(BUILD)

.PHONY: all clean
