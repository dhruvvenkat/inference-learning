# Compiler and flags
CXX      := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -Iinclude

# Directories
SRC_DIR := src
OBJ_DIR := obj

# Find all source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# DYNAMIC TARGET: Looks for the cpp file that isn't named matrix or utils, 
# and names the final executable after it (e.g., 'naive-multiply').
MAIN_SRC := $(filter-out $(SRC_DIR)/matrix.cpp, $(SRCS))
TARGET   := $(patsubst $(SRC_DIR)/%.cpp, %, $(MAIN_SRC))

# Default rule
all: $(TARGET)

# Link all compiled object files into the dynamically named executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

# Compile .cpp files into .o object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean

