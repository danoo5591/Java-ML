/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package juiciosnlp;
/**
 *
 * @author Hp envy
**/

import java.io.File;
import java.util.Map;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;


public class JuiciosNLP {
    public static String trainPath= "datasets/iris.csv";
    public static String testPath= "datasets/iris.csv";
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        Dataset data = FileHandler.loadDataset(new File(trainPath), 4, ",");
        Classifier knn = new KNearestNeighbors(5);
        knn.buildClassifier(data);
        Dataset dataForClassification = FileHandler.loadDataset(new File(testPath), 4, ",");
        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, dataForClassification);
        for (Object o : pm.keySet()) {
            System.out.println(o + ": " + pm.get(o).getAccuracy());
            System.out.println(o + ": " + pm.get(o).getPrecision());
            System.out.println(o + ": " + pm.get(o).getRecall());
            System.out.println(o + ": " + pm.get(o).getFMeasure());
        }
        
    }
    
}
