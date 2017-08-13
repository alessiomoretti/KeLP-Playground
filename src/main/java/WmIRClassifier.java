import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;

import java.util.HashMap;

public class WmIRClassifier {

    private float cSVM = 0.0f;

    private float precision = 0.0f;
    private float accuracy  = 0.0f;
    private float recall    = 0.0f;

    private float F1        = 0.0f;

    private HashMap<String, Float> F1Measures = new HashMap<String, Float>();
    private HashMap<String, Float> PrecisionMeasures = new HashMap<String, Float>();


    private HashMap<String, Float> RecallMeasures = new HashMap<String, Float>();


    private Classifier classifier = null;
    private OneVsAllLearning learner = null;

    public WmIRClassifier() { }

    public float getCSVM() {
        return cSVM;
    }

    public void setCSVM(float c_SVM) {
        cSVM = c_SVM;
    }

    public float getPrecision() {
        return precision;
    }

    public void setPrecision(float precision) {
        this.precision = precision;
    }

    public float getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(float accuracy) {
        this.accuracy = accuracy;
    }

    public float getRecall() {
        return recall;
    }

    public void setRecall(float recall) {
        this.recall = recall;
    }

    public float getF1() {
        return F1;
    }

    public void setF1(float f1) {
        F1 = f1;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public OneVsAllLearning getLearner() {
        return learner;
    }

    public void setLearner(OneVsAllLearning learner) {
        this.learner = learner;
    }

    public HashMap<String, Float> getF1Measures() {
        return F1Measures;
    }

    public HashMap<String, Float> getPrecisionMeasures() {
        return PrecisionMeasures;
    }

    public HashMap<String, Float> getRecallMeasures() {
        return RecallMeasures;
    }
}
