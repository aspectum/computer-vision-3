CXX ?= g++

CXXFLAGS += -Wall $(shell pkg-config --cflags --libs opencv)

all:  rectifiedSADmatcher

rectifiedSADmatcher: rectifiedSADmatcher.cpp;
	@$(CXX) $< -o $@ $(CXXFLAGS)

clean: ; 
	@rm -f *.o rectifiedSADmatcher

