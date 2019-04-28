python3 feature_builder.py CONLL_train.pos-chunk-name
python3 feature_builder.py CONLL_dev.pos-chunk
python3 feature_builder.py CONLL_test.pos-chunk

javac -cp ".:./maxent-3.0.0.jar:./trove.jar" MEtrain.java
javac -cp ".:./maxent-3.0.0.jar:./trove.jar" MEtag.java
java -cp ".:./maxent-3.0.0.jar:./trove.jar" MEtrain CONLL_train.pos-chunk-name-feature-enhanced model.MEtrain
java -cp ".:./maxent-3.0.0.jar:./trove.jar" MEtag CONLL_dev.pos-chunk-feature-enhanced model.MEtrain response.name
java -cp ".:./maxent-3.0.0.jar:./trove.jar" MEtag CONLL_test.pos-chunk-feature-enhanced model.MEtrain CONLL_test.name

python score.name.py