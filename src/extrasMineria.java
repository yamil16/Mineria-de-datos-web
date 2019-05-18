
public class extrasMineria {

	public extrasMineria() {
		// TODO Auto-generated constructor stub
	}
	
	
	//buildTrainTestClassificationModel(currentPath, multiple_instances2,multiple_instances2, funSimpleLogist);
	
	
			//Classifier classif=…
			/*
			//model 6 fold cross
			System.out.println("model 6 fold cros");
			Evaluation eTest =new Evaluation(stwv_instances2);
			eTest.crossValidateModel(funSimpleLogist,stwv_instances2,6,new Random());
			eTest.toSummaryString();
			eTest.toClassDetailsString();
			eTest.toMatrixString();
			//weka.core.SerializationHelper.write(currentPath,funSimpleLogist);
			buildCrossValidationModel(currentPath, stwv_instances2, funSimpleLogist);
		//	buildTrainTestClassificationModel(currentPath,stwv_instances2,eTest,model); 
			*/
			//split 66%
			
			//Classifier funSimpleLogist2 = new SimpleLogistic();
		/*	funSimpleLogist2.setBatchSize("100");
			funSimpleLogist2.setErrorOnProbabilities(false);
			funSimpleLogist2.setHeuristicStop(50);
			funSimpleLogist2.setMaxBoostingIterations(500);
			funSimpleLogist2.setNumBoostingIterations(0);
			funSimpleLogist2.setNumDecimalPlaces(2);
			funSimpleLogist2.setUseAIC(false);
			funSimpleLogist2.setUseCrossValidation(true);
			funSimpleLogist2.setWeightTrimBeta(0);
			System.out.println("model split");
			FilteredClassifier model = new FilteredClassifier();
	        model.setFilter(stwv2);
	        model.setClassifier(funSimpleLogist2);
	        */
	
	
	/*
	 *es lo mismo da el resultado
	model.buildClassifier(train);
	Evaluation eval = new Evaluation(train); 
	eval.evaluateModel(model, test); 
	System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	*/ 
	//model.buildClassifier(train);
	
	
	
	//buildTrainTestClassificationModel(currentPath,train,test,model);
	//buildTrainTestClassificationModel(currentPath,train,test,funSimpleLogist2);
	
	/*
	Classifier classif=new SimpleLogistic();
	classif.buildClassifier(train);//construyeel modelocon el training set
	Evaluation eTest2=new Evaluation(train);
	eTest2.evaluateModel(classif,test);//evalúacon el test set
	eTest.toSummaryString();
	eTest.toClassDetailsString();
	eTest.toMatrixString();
	*/
	
	/*
	//use training set
	System.out.println("model training set");
	Classifier inputMapped = new InputMappedClassifier();
	Classifier filteredClassifier = new FilteredClassifier();
	((FilteredClassifier)filteredClassifier).setFilter(stwv2);
	((FilteredClassifier)filteredClassifier).setClassifier(funSimpleLogist);
	((InputMappedClassifier)inputMapped).setClassifier(filteredClassifier);
	buildTrainTestClassificationModel(currentPath, stwv_instances2,stwv_instances2, inputMapped);
	*/
	//buildTrainTestClassificationModel(currentPath,stwv_instances2,instances2,funSimpleLogist);

	
	/*
	 * 
	 * 	Evaluation eTest =new Evaluation(multiple_instances2);//Acáno esnecesariohacerel build model!
	eTest.crossValidateModel(funSimpleLogist,multiple_instances2,6,new Random());
	//weka.core.SerializationHelper.write(currentPath,funSimpleLogist);
	
	buildCrossValidationModel(currentPath, multiple_instances2, funSimpleLogist);
	 * 
	int trainSize = (int) Math.round(stwv2.numInstances() * 0.8);
	int testSize = stwv2..numInstances() - trainSize;
	Instances train = new Instances(stwv2, 0, trainSize);
	Instances test = new Instances(stwv2, trainSize, testSize);
	model.buildClassifier(train);
	
	double percent = 66.0; 
	//Instances inst = ... // your full training set 
	int trainSize = (int) Math.round(stwv2.numInstances() * percent / 100); 
	int testSize = stwv2.numInstances() - trainSize; 
	Instances train = new Instances(stwv2, 0, trainSize); 
	Instances test = new Instances(stwv2, trainSize, testSize); 
	*/
	
	
	
	/*
	//InputMappedClassifier
	Classifier inputMapped = new InputMappedClassifier();
	Classifier filteredClassifier = new FilteredClassifier();
	((FilteredClassifier)filteredClassifier).setFilter(stwv2);
	((FilteredClassifier)filteredClassifier).setClassifier(new SMO());
	((InputMappedClassifier)inputMapped).setClassifier(filteredClassifier);
	buildTrainTestClassificationModel(currentPath, stwv_instances2,instances2, inputMapped);
	
	
	
	return instances2;
	*/

}
