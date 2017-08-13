import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.util.List;

public class WmIRQuestionClassificationExample {

    private final static int FOLDING = 5;
    private final static String CLASS = "HUM";
    private final static double[] C_POW_WEIGHT_ARRAY = {-2, -1, 0, 1, 2, 3};
    //private final static double[] C_POW_WEIGHT_ARRAY = {0};

    public static void main(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.println("Usage: kernel[lin| poly | tk | comb | comb-norm] c_svm");
            return;
        }

        // system statistics
        long start  = System.nanoTime();

        // reading the input parameters...
        String kernelType = args[0];
        // and preparing the
        float[] cSVMArray = new float[C_POW_WEIGHT_ARRAY.length];
        for (int c = 0; c < cSVMArray.length; c++) {
            cSVMArray[c] = (float) Math.pow(new Double(args[1]), (C_POW_WEIGHT_ARRAY[c]));
        }


        String trainingSetFilePath;
        String testSetFilePath;
        // importing from class resources the datasets
        try {
            trainingSetFilePath = WmIRQuestionClassificationExample.class.getClassLoader().getResource("datasets/qc_train.klp").getPath();
            testSetFilePath = WmIRQuestionClassificationExample.class.getClassLoader().getResource("datasets/qc_test.klp").getPath();
        } catch (NullPointerException e) {
            System.out.println("No valid path - no valid resources provided in the package");
            return;
        }

        // read the training and test dataset
        SimpleDataset trainingSet = new SimpleDataset();
        trainingSet.populate(trainingSetFilePath);
        //System.out.println("The training set is made of " + trainingSet.getNumberOfExamples() + " examples.");
        SimpleDataset testSet = new SimpleDataset();
        testSet.populate(testSetFilePath);
        //System.out.println("The test set is made of " + testSet.getNumberOfExamples() + " examples.");

        // print the number of train and test examples for each class
        //for (Label l : trainingSet.getClassificationLabels()) {
        //    System.out.println("Positive training examples for the class " + l.toString() + " "
        //            + trainingSet.getNumberOfPositiveExamples(l));
        //    System.out.println("Negative training examples for the class  " + l.toString() + " "
        //            + trainingSet.getNumberOfNegativeExamples(l));
        //}

        // calculating the size of the gram matrix to store all the examples
        int cacheSize = trainingSet.getNumberOfExamples() + testSet.getNumberOfExamples();

        // initialize the proper kernel function
        Kernel usedKernel = null;
        if (kernelType.equalsIgnoreCase("lin")) {
            String vectorRepresentationName = "bow";
            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            usedKernel = linearKernel;
        } else if (kernelType.equalsIgnoreCase("poly")) {
            String vectorRepresentationName = "bow";
            int exponent = 2;
            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel polynomialKernel = new PolynomialKernel(exponent, linearKernel);
            usedKernel = polynomialKernel;
        } else if (kernelType.equalsIgnoreCase("tk")) {
            String treeRepresentationName = "grct";
            float lambda = 0.4f;
            Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);
            usedKernel = tkgrct;
        } else if (kernelType.equalsIgnoreCase("comb")) {
            String vectorRepresentationName = "bow";
            String treeRepresentationName = "grct";
            float lambda = 0.4f;

            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);

            LinearKernelCombination combination = new LinearKernelCombination();
            combination.addKernel(1, linearKernel);
            combination.addKernel(1, tkgrct);
            usedKernel = combination;
        } else if (kernelType.equalsIgnoreCase("comb-norm")) {
            String vectorRepresentationName = "bow";
            String treeRepresentationName = "grct";
            float lambda = 0.4f;

            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
            Kernel treeKernel = new SubSetTreeKernel(lambda, treeRepresentationName);
            Kernel normalizedTreeKernel = new NormalizationKernel(treeKernel);

            LinearKernelCombination combination = new LinearKernelCombination();
            combination.addKernel(1, normalizedLinearKernel);
            combination.addKernel(1, normalizedTreeKernel);
            usedKernel = combination;
        } else {
            System.err.println("The specified kernel (" + kernelType + ") is not valid.");
        }

        // setting the cache to speed up the computations
        KernelCache cache = new FixIndexKernelCache(cacheSize);
        if (usedKernel != null) usedKernel.setKernelCache(cache);
        else {
            System.err.println("Not handled exception instantiating the Kernel / Kernel cache");
            return;
        }

        // learning and comparison steps
        WmIRClassifier classifier = COptimumLearning(cSVMArray, usedKernel, trainingSet, testSet);

        // computing elapsed time
        long elapsed = (System.nanoTime() - start) / 1000000000;

        // writing the learning algorithm and the kernel to file
        System.out.println("... writing learner and classifier in .klp files ...");
        JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
        serializer.writeValueOnFile(classifier.getLearner(), "ova_learning_algorithm.klp");
        serializer.writeValueOnFile(classifier.getClassifier(), "model_kernel-" + kernelType + "_cp" + classifier.getCSVM() + "_cn" + classifier.getCSVM() + ".klp");
        System.out.println("\n\n");

        // compensating the kelp library output
        System.out.flush();
        System.err.flush();

        // printing out results and statistics
        System.out.println(" CLASSIFIER");
        System.out.println("");
        System.out.println(" Kernel Type: " + kernelType);
        System.out.println(" C_SVM -> " + String.valueOf(classifier.getCSVM()));
        System.out.println(" ------------------------- ");
        System.out.println("      OVERALL METRICS      ");
        System.out.println(" ------------------------- ");
        System.out.println(" F1              " + String.valueOf(classifier.getF1()));
        System.out.println(" precision       " + String.valueOf(classifier.getPrecision()));
        System.out.println(" accuracy        " + String.valueOf(classifier.getAccuracy()));
        System.out.println(" recall          " + String.valueOf(classifier.getRecall()));
        System.out.println(" ------------------------- ");
        System.out.println("       CLASS '" + CLASS +    "' ");
        System.out.println(" ------------------------- ");
        System.out.println(" F1              " + String.valueOf(classifier.getF1Measures().get(CLASS)));
        System.out.println(" precision       " + String.valueOf(classifier.getPrecisionMeasures().get(CLASS)));
        System.out.println(" recall          " + String.valueOf(classifier.getRecallMeasures().get(CLASS)));
        System.out.println(" ------------------------- ");
        System.out.println("\n");
        System.out.println(" [SYSTEM] time elapsed: " + String.valueOf(elapsed) + " seconds");
    }


    private static WmIRClassifier COptimumLearning(float[] cSVMArray, Kernel kernel, SimpleDataset training, SimpleDataset validation) {

        WmIRClassifier[] classifiersArray = new WmIRClassifier[cSVMArray.length];


        // preparing the auxiliary variables
        int classifierIndex = 0;
        int maxClassifierIndex = 0;
        float maxClassifierScore = 0;

        // iterating over the array of possible weights to find the optimal learner
        // -> each learn will go with a N-fold cross validation to establish the best one
        for (float weight : cSVMArray) {
            // 1. instantiating the SVM learning algorithm
            BinaryCSvmClassification solver = new BinaryCSvmClassification();
            // setting the kernel
            solver.setKernel(kernel);
            // setting the weight
            solver.setC(weight);

            // 2. LEARNING step
            WmIRClassifier classifier = NFoldValidatedLearning(FOLDING, training, solver);

            // 3. COMPARISON step
            // 3.1 - validating using the validation set bundled with the application
            classifiersArray[classifierIndex] = EvaluateLearner(classifier.getLearner(), validation);
            classifiersArray[classifierIndex].setCSVM(weight);
            // 3.2 - comparison on F1 measure
            if (classifiersArray[classifierIndex].getF1Measures().get(CLASS) > maxClassifierScore) {
                maxClassifierScore = classifiersArray[classifierIndex].getF1Measures().get(CLASS);
                maxClassifierIndex = classifierIndex;
            }
            classifierIndex++;
        }

        // returning the chosen classifier
        return classifiersArray[maxClassifierIndex];
    }

    private static WmIRClassifier NFoldValidatedLearning(Integer folding, SimpleDataset trainingSet, BinaryCSvmClassification solver) {

        // instantiate the multi-class classifier that apply a One-vs-All schema
        OneVsAllLearning learner = new OneVsAllLearning();
        learner.setBaseAlgorithm(solver);
        learner.setLabels(trainingSet.getClassificationLabels());

        // --- N-FOLD CROSS VALIDATION  ---
        // 0. preparing the array of the classifiers and statistics
        WmIRClassifier[] classifiers = new WmIRClassifier[folding];

        // 1. SPLITTING the dataset - after random shuffle
        trainingSet.getShuffledDataset();
        SimpleDataset[] foldArray = trainingSet.nFolding(folding);

        // 2. TRAINING iterating over the folds
        for (int i = 0; i < folding; i++) {

            // getting validation set
            SimpleDataset validationSet = foldArray[i];

            // 2.1 - learning from the other folds
            for (int j = 0; j < folding; j++) {
                if (j != i)
                    learner.learn(foldArray[j]);
            }

            // 2.2 - producing the metrics and storing them along with the step by step classifiers
            classifiers[i] = EvaluateLearner(learner, validationSet);
        }

        // 3. METRIC average
        // instantiating the result classifier
        WmIRClassifier classifer = new WmIRClassifier();
        classifer.setLearner(learner);
        classifer.setClassifier(learner.getPredictionFunction());
        // averaging the produced metrics
        for (WmIRClassifier cls : classifiers) {
            classifer.setF1(classifer.getF1() + ((1/folding) * cls.getF1()));
            classifer.setRecall(classifer.getRecall() + ((1/folding) * cls.getRecall()));
            classifer.setAccuracy(classifer.getAccuracy() + ((1/folding) * cls.getAccuracy()));
            classifer.setPrecision(classifer.getPrecision() + ((1/folding) * cls.getPrecision()));
        }
        // returning the classifier object
        return classifer;
    }

    private static WmIRClassifier EvaluateLearner(OneVsAllLearning learner, SimpleDataset validationSet) {
        // F1 MEASURE - single class metric
        // note that precision = TP / (TP + FP)
        //          and recall = TP / (TP + FN)
        // then the F1 measures the test accuracy using an harmonic mean of precision and recall
        // F1 = 2 * (precision * recall) / (precision + recall)

        float TP = 0;
        float FP = 0;
        float FN = 0;
        float TN = 0;

        // initializing the result classifier
        WmIRClassifier wmIRClassifier =  new WmIRClassifier();

        // retrieving classifier from the learner
        Classifier classifier = learner.getPredictionFunction();

        // building the implicit confusion matrix from the metrics arrays
        List<Label> labels = validationSet.getClassificationLabels();
        int[] TPArray = new int[labels.size()];
        int[] FPArray = new int[labels.size()];
        int[] FNArray = new int[labels.size()];
        int[] TNArray = new int[labels.size()];

        // building the evaluator from kelp libraries
        //MulticlassClassificationEvaluator evaluator =
        // new MulticlassClassificationEvaluator(validationSet.getClassificationLabels());

        // classify examples - only according to the question class desired
        for (Example example : validationSet.getExamples()) {

            // classify the example
            ClassificationOutput predicted = classifier.predict(example);

            // adding to evaluator
            //evaluator.addCount(example, predicted);

            // computing metrics per label
            for (Label label : labels) {
                int labelIndex = labels.indexOf(label);
                if (example.isExampleOf(label)) {
                    // it is True Positive
                    if (predicted.getPredictedClasses().contains(label)) { TPArray[labelIndex]++; }
                    // or it is False Negative
                    else { FNArray[labelIndex]++; }
                } else {
                    // finally it is False Positive
                    if (predicted.getPredictedClasses().contains(label)) { FPArray[labelIndex]++; }
                    // or it is True Negative
                    else { TNArray[labelIndex]++; }
                }
            }
        }


        // computing the metrics over the implicit confusion matrix
        for (Label label : labels) {
            int i = labels.indexOf(label);
            TP += (float) TPArray[i];
            FP += (float) FPArray[i];
            FN += (float) FNArray[i];
            TN += (float) TNArray[i];

            // storing the F1 measures for every label
            float f1 = 2 * TPArray[i] / (float) ( 2 * TPArray[i] + FPArray[i] + FNArray[i]);
            wmIRClassifier.getF1Measures().put(label.toString(), f1);
            //wmIRClassifier.getF1Measures().put(label.toString(), evaluator.getF1For(label));
            // storing precision and recall for every label
            float prec = (float) TPArray[i] / (TPArray[i] + FPArray[i]);
            float rec  = (float) TPArray[i] / (TPArray[i] + FNArray[i]);
            wmIRClassifier.getPrecisionMeasures().put(label.toString(), prec);
            //wmIRClassifier.getPrecisionMeasures().put(label.toString(), evaluator.getPrecisionFor(label));
            wmIRClassifier.getRecallMeasures().put(label.toString(), rec);
            //wmIRClassifier.getRecallMeasures().put(label.toString(), evaluator.getRecallFor(label));

        }

        // calculating the overall precision and recall
        float precision = TP / (TP + FP);
        float recall    = TP / (TP + FN);

        // calculating the overall F1 measure
        float F1 =  2 * (precision * recall) / (precision + recall);

        // calculating overall (micro averaged) accuracy
        float accuracy = (TP + TN) / (TP + FP + FN + TN);

//        // DEBUG - comparing overall custom metrics with evaluator
//        System.out.println("RECALL " + String.valueOf(recall) + "(" + evaluator.getOverallRecall() + ")");
//        System.out.println("PRECISION " + String.valueOf(precision) + "(" + evaluator.getOverallPrecision() + ")");
//        System.out.println("F1 " + String.valueOf(F1) + "(" + evaluator.getOverallF1() + ")");


        wmIRClassifier.setPrecision(precision);
        wmIRClassifier.setAccuracy(accuracy);
        wmIRClassifier.setRecall(recall);
        wmIRClassifier.setF1(F1);

        wmIRClassifier.setClassifier(classifier);

        // returning a complete classifier object
        return wmIRClassifier;

    }

}
