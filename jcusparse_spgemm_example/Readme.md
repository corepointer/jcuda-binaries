## Compiling this little test program:

* Unpack the shared libs provided in the ubuntu1804 directory of this repository to some directory.

* Unpack the jar files from the other archive in ubuntu1804 to your maven directory M2_HOME (usually $HOME/.m2/repository)

* Compile & execute:

```
export M2_HOME=$HOME/.m2/repository
export LD_LIBRARY_PATH=../ubuntu1804:$LD_LIBRARY_PATH
export SAMPLE_CLASSPATH=$M2_HOME/org/jcuda/jcusparse/11.6.1/jcusparse-11.6.1.jar:$M2_HOME/org/jcuda/jcuda/11.6.1/jcuda-11.6.1.jar:.

javac -cp $M2_HOME/org/jcuda/jcusparse/11.6.1/jcusparse-11.6.1.jar:$M2_HOME/org/jcuda/jcuda/11.6.1/jcuda-11.6.1.jar jcuda/jcusparse/samples/JCusparseSgemmExample.java

java -cp $SAMPLE_CLASSPATH jcuda.jcusparse.samples.JCusparseSgemmExample

```
