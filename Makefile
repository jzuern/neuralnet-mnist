CC=g++
INC=-I/home/jzuern/Dropbox/develop/C++/Libraries/eigen/
CFLAGS=-c -Wall -std=c++11 $(INC)
LDFLAGS=
LDLIBS=
SOURCEPATH=./src
SOURCES=$(SOURCEPATH)/main.cpp \
	$(SOURCEPATH)/Data.cpp \
	$(SOURCEPATH)/NeuralNet.cpp


OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=nnet

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	    $(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

.cpp.o:
	    $(CC) $(CFLAGS) $< -o $@

clean:
			rm $(SOURCEPATH)/*.o
			rm nnet
