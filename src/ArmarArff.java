import java.io.IOException;
import java.io.File; 
import java.io.FileInputStream;
import java.io.FileOutputStream;

import java.nio.file.StandardCopyOption;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.Files;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import weka.attributeSelection.ASSearch;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.MultiStopwords;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.Tokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.StringToWordVector;

import com.google.gson.JsonIOException;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;




public class ArmarArff {
	static Set<String>StopWord;
	@SuppressWarnings("unused")
	private static Instances stwv_instances1;
	@SuppressWarnings("unused")
	private static Instances stwv_instances2;
	@SuppressWarnings("unused")
	private static String name;
	@SuppressWarnings("unused")
	private static String name2;
	public static void LeerArchivoStopWords(){
		final String rutaStopword= System.getProperty("user.dir")+"\\Stopwoard.txt";
		Set<String> StopWordAux=new HashSet<String>();		
		File archivo = null;
	    FileReader fr = null;
	    BufferedReader br = null;
	    try {
	         // Apertura del fichero y creacion de BufferedReader para poder
	         // hacer una lectura comoda (disponer del metodo readLine()).
	         archivo = new File (rutaStopword);
	         fr = new FileReader (archivo);
	         br = new BufferedReader(fr);
	         // Lectura del fichero
	         String linea;
	         while((linea=br.readLine())!=null)
	        	 StopWordAux.add(linea);
	         StopWord= new HashSet<String>(StopWordAux); 	            
	      }
	      catch(Exception e){
	         e.printStackTrace();
	      }finally{
	         // En el finally cerramos el fichero, para asegurarnos
	         // que se cierra tanto si todo va bien como si salta 
	         // una excepcion.
	         try{                    
	            if( null != fr ){   
	               fr.close();     
	            }                  
	         }catch (Exception e2){ 
	            e2.printStackTrace();
	         }
	      }		
	}
	

	public static String PreProcesamientoTexto(String texto,double cantpal, double cantpalExclamacion){
		String resultado="";
		Stemmer porter= new Stemmer();
		int posinic=0;
		String palabra="";
		String condicion="";
		String copiatexto=texto;
		while(posinic<copiatexto.length()){
			if(copiatexto.charAt(posinic)==(char)10){ //fin de linea
				posinic++;
			}
			else{
				int posfin=copiatexto.indexOf(':');
				copiatexto=copiatexto.substring(posfin+1, copiatexto.length());
			
				
				posfin=copiatexto.indexOf(':');				
				palabra=copiatexto.substring(0, posfin);
				copiatexto=copiatexto.substring(posfin+1, copiatexto.length());			
				
				
				
				posfin=copiatexto.indexOf((char)10);				
				condicion=copiatexto.substring(0, posfin);				
				copiatexto=copiatexto.substring(posfin+1, copiatexto.length());
				
			
				if( !StopWord.contains(palabra)){					
				
					if((condicion!="O") ||(condicion!="V") ||(condicion!="R")||(condicion!="E") ||(condicion!=",")||(condicion!="!")||(condicion!="G")||(condicion!="@")||(condicion!="#")||(condicion!="~")){ //Pronombres, VERBOS, ADVERBIOS, EMOTICONES, PUNTUACION, EXCLAMACION 
						cantpal++;
						
						for(int i=0;i<palabra.length();i++){
							if (Character.isLetter(palabra.charAt(i))){
								Character letra=Character.toLowerCase(palabra.charAt(i));
								if(letra.equals('$')==false){
									if ((Character.isDigit(letra))||(Character.isLetter(letra)))
										porter.add(letra);
									else
										porter.add(' ');
								}	
							}	
			                    
							 
						}
						//String [] aux = new String[1];
						//aux[0]=palabra;
						
						porter.stem();
						resultado=resultado +" "+porter.toString();
					/*
						porter.stem();
						
						String portstring=porter.toString();
						if(!portstring.contains("?"))
							resultado=resultado +" "+portstring;
							*/
					//	resultado=resultado +palabra;
					}
					else
					if(condicion=="!")
						cantpalExclamacion++;	
				}
				
			}	
				
		}			
		return resultado;
	}


	
	//Guardar las instances creadas como un .arff
	public static void saveArff(String ruta,Instances instancia) throws IOException{
		System.out.println("ruta "+ruta);
		ArffSaver saver=new ArffSaver();
		saver.setFile(new File(ruta));
		saver.setInstances(instancia);
		saver.writeBatch();
	}
		
	//Entrena,clasifica y guarda tanto el modelo como el resultado de la clasificación
	public static void buildTrainTestClassificationModel(String ruta,Instances instanciaEntrenamiento,Instances instanciaTest,Classifier clasificador) throws Exception{
		
		clasificador.buildClassifier(instanciaEntrenamiento); //construye el modelo con el training set
		Evaluation eTest=new Evaluation(instanciaEntrenamiento);
		eTest.evaluateModel(clasificador,instanciaTest); //evalúa con el test set
		
		weka.core.SerializationHelper.write(ruta+File.separator+"training-test.model", clasificador);
		saveStatistics(ruta+File.separator+"output-training-test.txt", eTest);
		
	}
	
	//Hace cross-validation, guarda el modelo generado y guarda los stats
	public static void buildCrossValidationModel(String ruta,Instances instancia,Classifier clasificador) throws Exception{
		Evaluation eTest = new Evaluation(instancia); //Acá no es necesario hacer el build model!
		eTest.crossValidateModel(clasificador, instancia, 10, new Random());
		
		weka.core.SerializationHelper.write(ruta+File.separator+"cross-validation.model", clasificador);
		saveStatistics(ruta+File.separator+"output-cross-validation.txt", eTest);
		
	}

	private static void saveStatistics(String ruta, Evaluation evaluacionTest)throws UnsupportedEncodingException, FileNotFoundException,IOException, Exception {
		BufferedWriter salida = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(ruta, false), "UTF8"));
		salida.write(evaluacionTest.toSummaryString(true));
		salida.newLine();
		salida.write(evaluacionTest.toClassDetailsString());
		salida.newLine();
		salida.write(evaluacionTest.toMatrixString());
		salida.close();
		
		System.out.println(evaluacionTest.toSummaryString());
		System.out.println(evaluacionTest.toClassDetailsString());
		System.out.println(evaluacionTest.toMatrixString());
	}
	public static void copiarArchivosJson(File directorio,String rutaOrigen) {
		final String rutaCarpeta= System.getProperty("user.dir")+"\\Data";
		File f = new File(rutaOrigen);		
			File[] ficheros = f.listFiles();
			
			for (int x=0;x<ficheros.length;x++){
		
				if (ficheros[x].isDirectory()){	
					
					copiarArchivosJson(ficheros[x],rutaOrigen+"\\"+ficheros[x].getName());
				}
				else{					
					Path FROM = Paths.get(rutaOrigen+"\\"+ficheros[x].getName());
			        Path TO = Paths.get(rutaCarpeta+"\\"+ficheros[x].getName());			       
			        try {
			        	 if (!Files.exists(TO)){
					            Files.createDirectories(TO); //si no anda hay que crear carpeta
			        	 } 
			        	 
					            Files.copy(FROM, TO, StandardCopyOption.REPLACE_EXISTING);
			        	      
					} catch (IOException e) {
						// TODO Auto-generated catch block
						System.out.println("SE ROMPE");
						System.out.println("SE ROMPE"+FROM.getFileName());
						System.out.println("SE ROMPE"+TO.getFileName());
						e.printStackTrace();
					}		      
				}
			}
		}
		
	public static void RecorrerInstancia2(String ruta, Instances instancia) throws JsonIOException, JsonSyntaxException, FileNotFoundException{
		File directorio = new File(ruta);
		int cont=0;
		if( directorio!=null){
			for(File archivo : directorio.listFiles()){			
				if(archivo.isFile()){ 
					if( archivo.getName().endsWith(".json")){
						cont++;
						JsonObject jsonObject = new JsonParser().parse(new BufferedReader(new InputStreamReader(new FileInputStream(archivo.getAbsolutePath())))).getAsJsonObject();
						System.out.println("cont: "+cont);
						String text;
						if(jsonObject.get("text").isJsonNull())
							text="";
						else{
							text = jsonObject.get("text").getAsString();
							//text=POSTagger.tagTweetStr(text);
							//text=PreProcesamientoTexto(text);					
						}	
						int favoritecount;		
						if(jsonObject.get("favorite_count").isJsonNull())
							favoritecount=-1;
						else				
							favoritecount= jsonObject.get("favorite_count").getAsInt();	
						String retweeted;
						if(jsonObject.get("retweeted").isJsonNull())
							retweeted="?";
						else
							retweeted=jsonObject.get("retweeted").getAsString();
						String favourite;
						if(jsonObject.get("favorited").isJsonNull())
							favourite="?";
						else
							favourite=jsonObject.get("favorited").getAsString();				
						double id;		
						if(jsonObject.get("id").isJsonNull())
							id=-1;
						else				
							id=jsonObject.get("id").getAsDouble();				
						double in_reply_to_status_id;
						if(jsonObject.get("in_reply_to_status_id").isJsonNull())
							in_reply_to_status_id=-1;
						else					
							in_reply_to_status_id=jsonObject.get("in_reply_to_status_id").getAsDouble();				
						@SuppressWarnings("unused")
						String lang = "?";
						if(jsonObject.get("lang").isJsonNull()) {
						} else
							lang=jsonObject.get("lang").getAsString();
						String created_at_msj;				
						if(jsonObject.get("created_at").isJsonNull())
							created_at_msj="?";
						else
							created_at_msj=jsonObject.get("created_at").getAsString();			    
						String created_at_cuenta;
						if(jsonObject.getAsJsonObject("user").get("created_at").isJsonNull())
							created_at_cuenta="?";
						else
							created_at_cuenta=jsonObject.getAsJsonObject("user").get("created_at").getAsString();
						String verified;
						if(jsonObject.getAsJsonObject("user").get("verified").isJsonNull())
							verified="?";
						else
							verified=jsonObject.getAsJsonObject("user").get("verified").getAsString();
						@SuppressWarnings("unused")
						String time_zone;
						if(jsonObject.getAsJsonObject("user").get("time_zone").isJsonNull())
							 time_zone="?";
						else
							 time_zone=jsonObject.getAsJsonObject("user").get("time_zone").getAsString();				
						int friends_count;		
						if(jsonObject.getAsJsonObject("user").get("friends_count").isJsonNull())
							friends_count=-1;
						else				
							 friends_count=jsonObject.getAsJsonObject("user").get("friends_count").getAsInt();			
						int followers_count;		
						if(jsonObject.getAsJsonObject("user").get("followers_count").isJsonNull())
							followers_count=-1;
						else				
							followers_count=jsonObject.getAsJsonObject("user").get("followers_count").getAsInt();
						int listed_count;		
						if(jsonObject.getAsJsonObject("user").get("listed_count").isJsonNull())
							listed_count=-1;
						else				
							listed_count=jsonObject.getAsJsonObject("user").get("listed_count").getAsInt();
						int retweet_count;		
						if(jsonObject.get("retweet_count").isJsonNull())
							retweet_count=-1;
						else				
							retweet_count= jsonObject.get("retweet_count").getAsInt();	
						//if(jsonObject.getAsJsonObject("user").get("name").isJsonNull())
							//name2="?";
						//else
							//name2=jsonObject.getAsJsonObject("user").get("name").getAsString();
						int statuses_count;		
						if(jsonObject.getAsJsonObject("user").get("statuses_count").isJsonNull())
							statuses_count=-1;
						else				
							statuses_count=jsonObject.getAsJsonObject("user").get("statuses_count").getAsInt();
						String description;
						if((jsonObject.getAsJsonObject("user").get("description").isJsonNull())||(jsonObject.getAsJsonObject("user").get("description").getAsString()==""))
							description="?";
						else{
							description=jsonObject.getAsJsonObject("user").get("description").getAsString();
							if(description.length()>1){				
								//description=POSTagger.tagTweetStr(description);				
								//description=PreProcesamientoTexto(description);				
							}
						}					
						double [] vals = new double[instancia.numAttributes()];	
						System.out.println("nombreArchivo getparen "+archivo.getParent());
						System.out.println("nombreArchivo getpath "+archivo.getPath());
						if(archivo.getPath().contains("non-rumours"))
							vals[0]=0;
						else
							vals[0]=1;
						vals[1] = instancia.attribute(1).addStringValue(text);
						vals[2] =favoritecount;
						if(retweeted=="true")
							vals[3]=1;
						else
							vals[3]=0;
						if(favourite=="true")
							vals[4]=1;
						else
							vals[4]=0;
						if(id==in_reply_to_status_id)
							vals[5]=1;
						else
							vals[5]=0;
						if(verified=="true")
							vals[6]=1;
						else
							vals[6]=0;	
						vals[7]=friends_count;
						vals[8]=followers_count;
						vals[9]=listed_count;
						vals[10]=retweet_count;
						vals[11]=statuses_count;
						
						@SuppressWarnings("deprecation")
						Date date2_created_at_msj= new Date(created_at_msj);			
						@SuppressWarnings("deprecation")
						Date date2_created_at_cuenta= new Date(created_at_cuenta);				
						double milisegundosDia=24*60*60*1000;
						double diferencia=(date2_created_at_msj.getTime()-date2_created_at_cuenta.getTime())/milisegundosDia;
						vals[12]=diferencia;			
						Instance i = new DenseInstance(1.0,vals); //crear la instancia con los valores ya cargados
						instancia.add(i); //agregar la instancia
					}
					else// sino es archivo json
					{
						if (archivo.isDirectory()){
						String ruta2=ruta+File.pathSeparator+archivo.getName();
						System.out.println("ruta2 "+ruta2);
						RecorrerInstancia2(ruta2,  instancia);
						}
					}
				}
				else{//si es directorio
					if (archivo.isDirectory()){
						String ruta2=ruta+"\\"+archivo.getName();
						System.out.println("ruta2 "+ruta2);
						RecorrerInstancia2(ruta2,  instancia);
					}
					
				}
			}
		}
	}
	
	public static Instances generarInstancia2(String ruta) throws JsonIOException, JsonSyntaxException, FileNotFoundException{
		List<Attribute> atributo = new ArrayList<Attribute>(); //creamos la lista de atributos!		
		List<String> rumor = new ArrayList<String>(); //creamos la clase
		rumor.add("yes");
		rumor.add("no");		
		List<String> respuestatweet = new ArrayList<String>(); //creamos la clase
		respuestatweet.add("yes");
		respuestatweet.add("no");
		atributo.add(new Attribute("1_es_rumor",rumor)); //creación de atributo nominal
		atributo.add(new Attribute("2_texto_tweet",true)); //creación de atributo string
		atributo.add(new Attribute("3_cantidad_favorito"));
		atributo.add(new Attribute("4_fue_retweed"));
		atributo.add(new Attribute("5_es_favorito"));
		atributo.add(new Attribute("6_respuesta_tweet",respuestatweet)); //usa id y in_reply_to_status_id 				
		atributo.add(new Attribute("7_cuenta_verificada")); //usuario verificado	
		atributo.add(new Attribute("8_cantidad_amigos")); //nro de usuarios que esta cuenta sigue
		atributo.add(new Attribute("9_cantidad_seguidores")); //nro seguidores que siguen a la cuenta
		atributo.add(new Attribute("10_cantidad_listed"));
		atributo.add(new Attribute("11_cantidad_retweet"));		
		atributo.add(new Attribute("12_cantidad_status"));	//nro
		atributo.add(new Attribute("13_diferencia_fecha"));				
		Instances instancia = new Instances("relations",(ArrayList<Attribute>) atributo,1); //se crea el Instances con los atributos definidos!
		instancia.setClassIndex(0); //setear el índice de la clase		
		RecorrerInstancia2(ruta,  instancia);
		return instancia;
	}

	public static Instances GenerarArff2(String currentPath) throws Exception{
		Instances instances2 = generarInstancia2(currentPath); //current working directory
		System.out.println("inst" +instances2.numAttributes());
		saveArff(currentPath+File.separator+"original_instances2.arff", instances2);
		System.out.println(instances2);
		
	
		ClassBalancer balancer2 = new ClassBalancer();
		balancer2.setInputFormat(instances2);
		Instances balanced_instances2 = Filter.useFilter(instances2, balancer2);
		saveArff(currentPath+File.separator+"balanced_instances2.arff", balanced_instances2);
		//	normalizar.setInputFormat(instances2);
		LovinsStemmer stem= new LovinsStemmer();
		
		Filter stwv2 =new StringToWordVector();
		stwv2.setInputFormat(instances2); //Acá se puede configurar stemmer, stopword removal, ponderación de atributos...				
		//((StringToWordVector) stwv).setTFTransform(true);
		((StringToWordVector) stwv2).setIDFTransform(true);
		//Tokenizer tk2=new NGramTokenizer();
		Tokenizer tk2=new WordTokenizer();
		 //CharacterDelimitedTokenizer cd= new ;
		 //tk.setOptions(new CharacterDelimitedTokenizer().setDelimiters(3));
		
		
		//Rainbow StopWordRainbox= new Rainbow();
		MultiStopwords StopWordMulti=new MultiStopwords(); 
		((StringToWordVector) stwv2).setTokenizer(tk2);		
		((StringToWordVector) stwv2).setNormalizeDocLength(new SelectedTag(weka.filters.unsupervised.attribute.StringToWordVector.FILTER_NORMALIZE_ALL, weka.filters.unsupervised.attribute.StringToWordVector.TAGS_FILTER));		
		//((StringToWordVector) stwv2).setStopwordsHandler(StopWordRainbox);
		((StringToWordVector) stwv2).setStopwordsHandler(StopWordMulti);
		((StringToWordVector) stwv2).setStemmer(stem);;
		
		Instances stwv_instances2 = Filter.useFilter(instances2, stwv2);
		saveArff(currentPath+File.separator+"stwv_instances2.arff", stwv_instances2);
			
		
		
		Filter infoGain2 = new AttributeSelection();
		infoGain2.setInputFormat(stwv_instances2);
		((weka.filters.supervised.attribute.AttributeSelection)infoGain2).setEvaluator(new InfoGainAttributeEval());
		ASSearch ranker2_1 = new Ranker();
		((Ranker) ranker2_1).setNumToSelect((int)(stwv_instances2.numAttributes()*0.5));
		((weka.filters.supervised.attribute.AttributeSelection)infoGain2).setSearch(ranker2_1);
		Instances infogain_instances2 = Filter.useFilter(stwv_instances2, infoGain2);
		saveArff(currentPath+File.separator+"infogain_instances2.arff", infogain_instances2);
		
		/*
		//2
		Filter ComponentePrincipal2= new AttributeSelection();
		ComponentePrincipal2.setInputFormat(stwv_instances2);
		((weka.filters.supervised.attribute.AttributeSelection) ComponentePrincipal2).setEvaluator(new weka.attributeSelection.PrincipalComponents());
		ASSearch ranker2_2 = new Ranker();
		((Ranker) ranker2_2).setNumToSelect((int)(stwv_instances2.numAttributes()*0.5));
		((weka.filters.supervised.attribute.AttributeSelection) ComponentePrincipal2).setSearch(ranker2_2);
		Instances ComponentePrincipal_instances2 = Filter.useFilter(stwv_instances2, ComponentePrincipal2);
		saveArff(currentPath+File.separator+"ComponentePrincipal_instances2.arff", ComponentePrincipal_instances2);
		//3
		Filter ReliefAtribEval2= new AttributeSelection();
		ReliefAtribEval2.setInputFormat(stwv_instances2);
		((weka.filters.supervised.attribute.AttributeSelection) ReliefAtribEval2).setEvaluator(new weka.attributeSelection.ReliefFAttributeEval());
		ASSearch ranker2_3 = new Ranker();
		((Ranker) ranker2_3).setNumToSelect((int)(stwv_instances2.numAttributes()*0.5));
		((weka.filters.supervised.attribute.AttributeSelection) ReliefAtribEval2).setSearch(ranker2_3);
		Instances ReliefAtribEval_instances2 = Filter.useFilter(stwv_instances2, ReliefAtribEval2);
		saveArff(currentPath+File.separator+"ReliefAtribEvall_instances.arff", ReliefAtribEval_instances2);
		*/
		
	
		//********* Multiple Filters!
		MultiFilter multiFilter2 = new MultiFilter();
		multiFilter2.setInputFormat(instances2);
		multiFilter2.setFilters(new Filter[]{stwv2,infoGain2});
		Instances multiple_instances2 = Filter.useFilter(instances2, multiFilter2);
		saveArff(currentPath+File.separator+"multi_instances2.arff", multiple_instances2);
		
		/*
		//nuevo yam combo
		MultiFilter multiFilterGeneral = new MultiFilter();
		multiFilterGeneral.setInputFormat(instances2);
		multiFilterGeneral.setFilters(new Filter[]{stwv2,infoGain2,ReliefAtribEval2,ComponentePrincipal2});
		Instances multiple_instances_general = Filter.useFilter(instances2, multiFilterGeneral);
		saveArff(currentPath+File.separator+"multiple_instances_general2.arff", multiple_instances_general);
		
		MultiFilter multiFilterComponentePrincipal = new MultiFilter();
		multiFilterComponentePrincipal.setInputFormat(instances2);
		multiFilterComponentePrincipal.setFilters(new Filter[]{stwv2,ComponentePrincipal2});
		Instances multiple_instances_ComponentePrincipal = Filter.useFilter(instances2, multiFilterComponentePrincipal);
		saveArff(currentPath+File.separator+"multiple_instances_ComponentePrincipal2.arff", multiple_instances_ComponentePrincipal);
		
		MultiFilter multiFilterReliefAtribEval = new MultiFilter();
		multiFilterReliefAtribEval.setInputFormat(instances2);
		multiFilterReliefAtribEval.setFilters(new Filter[]{stwv2,ReliefAtribEval2});
		Instances multiple_instances_ReliefAtribEval= Filter.useFilter(instances2, multiFilterReliefAtribEval);
		saveArff(currentPath+File.separator+"multiple_instances_ReliefAtribEval2.arff", multiple_instances_ReliefAtribEval);
		*/
		
		//Ejemplos de Classification!
		//Classifier smo = new SMO(); //Ojo!! Se puede parametrizar!
		
		//Classifier funSimpleLogist = new SimpleLogistic();
		
		
		/* SPILT 66%
		double percent = 66.0;	
		int trainSize = (int) Math.round(stwv_instances2.numInstances() * percent / 100); 
		System.out.println("model trainSize "+trainSize);
		int testSize =stwv_instances2.numInstances() - trainSize; 
		System.out.println("model testSize "+testSize);
		Instances train = new Instances(stwv_instances2, 0, trainSize); 
		Instances test = new Instances(stwv_instances2, trainSize, testSize); 

		System.out.println("test.numInstances "+test.numInstances());
		System.out.println("trainSize.numInstances "+train.numInstances());
		//nuevo
		
		Classifier funSimpleLogist2 = new SimpleLogistic();
		funSimpleLogist2.buildClassifier(train);
		Evaluation eTest = new Evaluation(train);
		eTest.evaluateModel(funSimpleLogist2, test);
		 String strSummary = eTest.toSummaryString();
		 System.out.println("split "+strSummary);
		*/
		 //cros
		
		//Instances trainCrossValidation = new Instances(stwv_instances2); 
		Instances trainCrossValidation = new Instances(multiple_instances2);
		Classifier funSimpleLogistCrossValidation = new SimpleLogistic();
		Evaluation eTestCrossValidation = new Evaluation(trainCrossValidation);
		eTestCrossValidation.crossValidateModel(funSimpleLogistCrossValidation,trainCrossValidation,6,new Random());
		 String strSummaryCrossValidation = eTestCrossValidation.toSummaryString();
		 System.out.println(strSummaryCrossValidation);
		 
	
		File f;
		f = new File(currentPath+File.separator+"Resultados modelo creado con cross validation Arff2.txt");
		FileWriter w = new FileWriter(f);
		BufferedWriter bw = new BufferedWriter(w);	
		PrintWriter wr = new PrintWriter(bw);  
		wr.write(strSummaryCrossValidation);
		wr.close();
		bw.close();
		
			 
		
		// buildCrossValidationModel(currentPath, trainCrossValidation, funSimpleLogist);
		 //weka.core.SerializationHelper.write(currentPath,funSimpleLogist);
		
	
		return multiple_instances2;
		
	
	}
	
	
	public static void recorrerInstanciaPrincipal(String ruta, Instances instancia) throws JsonIOException, JsonSyntaxException, FileNotFoundException{
		File directorio = new File(ruta);
		int cont=0;
		
		if( directorio!=null){
			for(File archivo : directorio.listFiles()){			
				if(archivo.isFile()){ 
					if( archivo.getName().endsWith(".json")){
						System.out.println("nombre archivo json "+archivo.getName());
						
						double cantpal=0;
						double cantpalExclamacion=0;
						cont++;
						JsonObject jsonObject = new JsonParser().parse(new BufferedReader(new InputStreamReader(new FileInputStream(archivo.getAbsolutePath())))).getAsJsonObject();
						System.out.println("cont: "+cont);
						String lang;
						if(jsonObject.get("lang").isJsonNull())
						lang="?";
						else
							lang=jsonObject.get("lang").getAsString();	
						
						String text;
						if(jsonObject.get("text").isJsonNull())
							text="?";
						else{
							text = jsonObject.get("text").getAsString();
							if(text.length()>1){
								text=POSTagger.tagTweetStr(text);
								text=PreProcesamientoTexto(text,cantpal,cantpalExclamacion);
								text=text.replaceAll("$", " ");
							
									
							}
						}	
						int favoritecount;		
						if(jsonObject.get("favorite_count").isJsonNull())
							favoritecount=-1;
						else				
							favoritecount= jsonObject.get("favorite_count").getAsInt();	
						String retweeted;
						if(jsonObject.get("retweeted").isJsonNull())
						retweeted="?";
						else
							retweeted=jsonObject.get("retweeted").getAsString();
						String favourite;
						if(jsonObject.get("favorited").isJsonNull())
						favourite="?";
						else
							favourite=jsonObject.get("favorited").getAsString();
						double id;		
						if(jsonObject.get("id").isJsonNull())
							id=-1;
						else				
							id=jsonObject.get("id").getAsDouble();	
						double in_reply_to_status_id;
						if(jsonObject.get("in_reply_to_status_id").isJsonNull())
							in_reply_to_status_id=-1;
						else					
							in_reply_to_status_id=jsonObject.get("in_reply_to_status_id").getAsDouble();			
							
						String created_at_msj;			
						if(jsonObject.get("created_at").isJsonNull())
						created_at_msj="?";
						else
							created_at_msj=jsonObject.get("created_at").getAsString();			
						String created_at_cuenta;
						if(jsonObject.getAsJsonObject("user").get("created_at").isJsonNull())
							created_at_cuenta="?";
						else
							created_at_cuenta=jsonObject.getAsJsonObject("user").get("created_at").getAsString();
						String verified;
						if(jsonObject.getAsJsonObject("user").get("verified").isJsonNull())
							verified="?";
						else
							verified=jsonObject.getAsJsonObject("user").get("verified").getAsString();
						String time_zone;
						if(jsonObject.getAsJsonObject("user").get("time_zone").isJsonNull())
							time_zone="?";
						else{
							 time_zone=jsonObject.getAsJsonObject("user").get("time_zone").getAsString();
							// time_zone=POSTagger.tagTweetStr(time_zone);				
							// time_zone=PreProcesamientoTexto(time_zone,cantpal,cantpalExclamacion);
						}
						
						
						
						int friends_count;		
						if(jsonObject.getAsJsonObject("user").get("friends_count").isJsonNull())
							friends_count=-1;
						else				
							 friends_count=jsonObject.getAsJsonObject("user").get("friends_count").getAsInt();			
						int followers_count;		
						if(jsonObject.getAsJsonObject("user").get("followers_count").isJsonNull())
							followers_count=-1;
						else				
							followers_count=jsonObject.getAsJsonObject("user").get("followers_count").getAsInt();
						int listed_count;		
						if(jsonObject.getAsJsonObject("user").get("listed_count").isJsonNull())
							listed_count=-1;
						else				
							listed_count=jsonObject.getAsJsonObject("user").get("listed_count").getAsInt();
						int retweet_count;		
						if(jsonObject.get("retweet_count").isJsonNull())
							retweet_count=-1;
						else				
							retweet_count= jsonObject.get("retweet_count").getAsInt();	
						name = "";
						if(jsonObject.getAsJsonObject("user").get("name").isJsonNull())
						name="?";
						else
							name=jsonObject.getAsJsonObject("user").get("name").getAsString();
						int statuses_count;		
						if(jsonObject.getAsJsonObject("user").get("statuses_count").isJsonNull())
							statuses_count=-1;
						else				
							statuses_count=jsonObject.getAsJsonObject("user").get("statuses_count").getAsInt();
						String description;
						if((jsonObject.getAsJsonObject("user").get("description").isJsonNull())||(jsonObject.getAsJsonObject("user").get("description").getAsString()==""))
						description="?";
						else{
							description=jsonObject.getAsJsonObject("user").get("description").getAsString();
							if(description.length()>1){				
							description=POSTagger.tagTweetStr(description);				
							description=PreProcesamientoTexto(description,cantpal,cantpalExclamacion);
							description=description.replaceAll("$", " ");
							}
						}				
						String source=jsonObject.get("source").getAsString();			
						double [] vals = new double[instancia.numAttributes()];		
						vals[0] = 0; //índice de la clase a asignar. Acá HARDCODEADA LA CLASE.
						if(archivo.getPath().contains("non-rumours"))
							vals[0]=0;
						else
							vals[0]=1;
						vals[1] = instancia.attribute(1).addStringValue(text);
						vals[2] =favoritecount;
						if(retweeted=="true")
							vals[3]=1;
						else
							vals[3]=0;
						if(favourite=="true")
							vals[4]=1;
						else
							vals[4]=0;	
						vals[5]=id;
						vals[6]=in_reply_to_status_id;
						if(lang=="en") //lo tomo asi para cuantificar el idioma en ingles
							vals[7]=1;
						else
							vals[7]=0;	
						@SuppressWarnings("deprecation")
						Date date2_created_at_msj= new Date(created_at_msj);			
						@SuppressWarnings("deprecation")
						Date date2_created_at_cuenta= new Date(created_at_cuenta);
						if(verified=="true")
							vals[8]=1;
						else
							vals[8]=0;	
						vals[9]=instancia.attribute(9).addStringValue(time_zone);
						vals[10]=friends_count;
						vals[11]=followers_count;
						vals[12]=listed_count;			
						vals[13]=retweet_count;
						vals[14]=statuses_count;
						vals[15]=instancia.attribute(15).addStringValue(description);		
						source=source.toLowerCase();
						if(source.contains("web"))
							vals[16]=1;
						else
						if(source.contains("iphone"))
							vals[16]=0;
						else
							vals[16]=-1;			
						double milisegundosDia=24*60*60*1000;
						System.out.println("tiempo msj "+date2_created_at_msj.getTime()+" tiempo cta "+date2_created_at_cuenta.getTime());			
						double diferencia=(date2_created_at_msj.getTime()-date2_created_at_cuenta.getTime())/milisegundosDia;			
							vals[17]=diferencia;
						
						vals[18]=cantpal;
						vals[19]=cantpalExclamacion;
							
						Instance i = new DenseInstance(1.0,vals); //crear la instancia con los valores ya cargados
						instancia.add(i); //agregar la instancia
					}
					else// sino es archivo json
					{
						if (archivo.isDirectory()){
						String ruta2=ruta+File.pathSeparator+archivo.getName();
						System.out.println("ruta2 "+ruta2);
						recorrerInstanciaPrincipal(ruta2,  instancia);
						}
					}
				}
				else{//si es directorio
					if (archivo.isDirectory()){
						String ruta2=ruta+"\\"+archivo.getName();
						System.out.println("ruta2 "+ruta2);
						recorrerInstanciaPrincipal(ruta2,  instancia);
					}
					
				}
			}
		}
	}
	

	public static Instances generarInstanciaPrincipal(String ruta) throws JsonIOException, JsonSyntaxException, FileNotFoundException{
		List<Attribute> atributo = new ArrayList<Attribute>(); //creamos la lista de atributos!		
		List<String> rumor = new ArrayList<String>(); //creamos la clase
		rumor.add("yes");
		rumor.add("no");		
		atributo.add(new Attribute("1_es_rumour",rumor)); //creación de atributo nominal
		atributo.add(new Attribute("2_texto_tweet",true)); //creación de atributo string
		atributo.add(new Attribute("3_cantidad_favorito"));
		atributo.add(new Attribute("4_fue_retweed"));
		atributo.add(new Attribute("5_es_favorito"));
		atributo.add(new Attribute("6_identificacion_tweet")); //¿¿tendria que ser nro??
		atributo.add(new Attribute("7_es_una_respuesta_al_tweet_id")); //¿¿tendria que ser nro?? Para saber si es una respuesta a un tweet si es una reaccion es igual al id_str del msj fuente
		atributo.add(new Attribute("8_lenguaje_ingles")); //lenguaje	
		atributo.add(new Attribute("9_cuenta_verificada")); //usuario verificado
		atributo.add(new Attribute("10_tiempo_zona",true));
		atributo.add(new Attribute("11_cantidad_amigos"));
		atributo.add(new Attribute("12_cantidad_seguidores"));
		atributo.add(new Attribute("13_cantidad_listed"));
		atributo.add(new Attribute("14_cantidad_retweet"));
		atributo.add(new Attribute("15_cantidad_status"));
		atributo.add(new Attribute("16_description",true));
		atributo.add(new Attribute("17_origen_envio_msj_dispositivo"));
		atributo.add(new Attribute("18_diferencia_fecha"));	
		atributo.add(new Attribute("19_cantidad_palabras_texto"));
		atributo.add(new Attribute("20_cantidad_palabras_exclamacion"));		
		Instances instancia = new Instances("relations",(ArrayList<Attribute>) atributo,1); //se crea el Instances con los atributos definidos!
		instancia.setClassIndex(0); //setear el índice de la clase		
		recorrerInstanciaPrincipal(ruta,  instancia);
		return instancia;
	}
	

	
	public static Instances GenerarArffPrincipal(String currentPath) throws Exception{
		Instances instances2 = generarInstanciaPrincipal(currentPath); //current working directory
		System.out.println("inst" +instances2.numAttributes());
		saveArff(currentPath+File.separator+"original_instances1.arff", instances2);
		System.out.println(instances2);
		
		
		ClassBalancer balancer2 = new ClassBalancer();
		balancer2.setInputFormat(instances2);
		Instances balanced_instances2 = Filter.useFilter(instances2, balancer2);
		saveArff(currentPath+File.separator+"balanced_instances1.arff", balanced_instances2);
		
		Filter stwv2 =new StringToWordVector();
		stwv2.setInputFormat(instances2); //Acá se puede configurar stemmer, stopword removal, ponderación de atributos...				
		//((StringToWordVector) stwv).setTFTransform(true);
		((StringToWordVector) stwv2).setIDFTransform(true);
		Tokenizer tk2=new NGramTokenizer();
		 //CharacterDelimitedTokenizer cd= new ;
		 //tk.setOptions(new CharacterDelimitedTokenizer().setDelimiters(3));		 
		((StringToWordVector) stwv2).setTokenizer(tk2);		
		((StringToWordVector) stwv2).setNormalizeDocLength(new SelectedTag(weka.filters.unsupervised.attribute.StringToWordVector.FILTER_NORMALIZE_ALL, weka.filters.unsupervised.attribute.StringToWordVector.TAGS_FILTER));		
		Instances stwv_instances2 = Filter.useFilter(instances2, stwv2);
		saveArff(currentPath+File.separator+"stwv_instances1.arff", stwv_instances2);
	
		
		
		Filter infoGain2 = new AttributeSelection();
		infoGain2.setInputFormat(stwv_instances2);
		((weka.filters.supervised.attribute.AttributeSelection)infoGain2).setEvaluator(new InfoGainAttributeEval());
		ASSearch ranker2_1 = new Ranker();
		((Ranker) ranker2_1).setNumToSelect((int)(stwv_instances2.numAttributes()*0.5));
		((weka.filters.supervised.attribute.AttributeSelection)infoGain2).setSearch(ranker2_1);
		Instances infogain_instances2 = Filter.useFilter(stwv_instances2, infoGain2);
		saveArff(currentPath+File.separator+"infogain_instances1.arff", infogain_instances2);
	
		
		/*
		//2
		Filter ComponentePrincipal2= new AttributeSelection();
		ComponentePrincipal2.setInputFormat(stwv_instances2);
		((weka.filters.supervised.attribute.AttributeSelection) ComponentePrincipal2).setEvaluator(new weka.attributeSelection.PrincipalComponents());
		ASSearch ranker2_2 = new Ranker();
		((Ranker) ranker2_2).setNumToSelect((int)(stwv_instances2.numAttributes()*0.5));
		((weka.filters.supervised.attribute.AttributeSelection) ComponentePrincipal2).setSearch(ranker2_2);
		Instances ComponentePrincipal_instances2 = Filter.useFilter(stwv_instances2, ComponentePrincipal2);
		saveArff(currentPath+File.separator+"ComponentePrincipal_instances1.arff", ComponentePrincipal_instances2);
			*/	
		//3
		Filter ReliefAtribEval2= new AttributeSelection();
		ReliefAtribEval2.setInputFormat(stwv_instances2);
		((weka.filters.supervised.attribute.AttributeSelection) ReliefAtribEval2).setEvaluator(new weka.attributeSelection.ReliefFAttributeEval());
		ASSearch ranker2_3 = new Ranker();
		((Ranker) ranker2_3).setNumToSelect((int)(stwv_instances2.numAttributes()*0.5));
		((weka.filters.supervised.attribute.AttributeSelection) ReliefAtribEval2).setSearch(ranker2_3);
		Instances ReliefAtribEval_instances2 = Filter.useFilter(stwv_instances2, ReliefAtribEval2);
		saveArff(currentPath+File.separator+"ReliefAtribEvall_instances1.arff", ReliefAtribEval_instances2);

	
		//********* Multiple Filters!
		/*
		MultiFilter multiFilter2 = new MultiFilter();
		multiFilter2.setInputFormat(instances2);
		multiFilter2.setFilters(new Filter[]{stwv2,infoGain2});
		Instances multiple_instances2 = Filter.useFilter(instances2, multiFilter2);
		saveArff(currentPath+File.separator+"multi_instances1.arff", multiple_instances2);
		*/
		/*
		//nuevo yam combo
		MultiFilter multiFilterGeneral = new MultiFilter();
		multiFilterGeneral.setInputFormat(instances2);
		multiFilterGeneral.setFilters(new Filter[]{stwv2,infoGain2,ReliefAtribEval2,ComponentePrincipal2});
		Instances multiple_instances_general = Filter.useFilter(instances2, multiFilterGeneral);
		saveArff(currentPath+File.separator+"multiple_instances_general1.arff", multiple_instances_general);
		
		MultiFilter multiFilterComponentePrincipal = new MultiFilter();
		multiFilterComponentePrincipal.setInputFormat(instances2);
		multiFilterComponentePrincipal.setFilters(new Filter[]{stwv2,ComponentePrincipal2});
		Instances multiple_instances_ComponentePrincipal = Filter.useFilter(instances2, multiFilterComponentePrincipal);
		saveArff(currentPath+File.separator+"multiple_instances_ComponentePrincipal1.arff", multiple_instances_ComponentePrincipal);
		*/
		
		/*
		MultiFilter multiFilterReliefAtribEval = new MultiFilter();
		multiFilterReliefAtribEval.setInputFormat(instances2);
		multiFilterReliefAtribEval.setFilters(new Filter[]{stwv2,ReliefAtribEval2});
		Instances multiple_instances_ReliefAtribEval= Filter.useFilter(instances2, multiFilterReliefAtribEval);
		saveArff(currentPath+File.separator+"multiple_instances_ReliefAtribEval1.arff", multiple_instances_ReliefAtribEval);
		*/
		MultiFilter multiFilterGeneral = new MultiFilter();
		multiFilterGeneral.setInputFormat(instances2);
		multiFilterGeneral.setFilters(new Filter[]{stwv2,infoGain2,ReliefAtribEval2});
		Instances multiple_instances_general = Filter.useFilter(instances2, multiFilterGeneral);
		saveArff(currentPath+File.separator+"multiple_instances_general1.arff", multiple_instances_general);
		
		
		
		//Instances trainCrossValidation = new Instances(stwv_instances2); 
		Instances trainCrossValidation = new Instances(multiple_instances_general);
		Classifier funSimpleLogistCrossValidation = new SimpleLogistic();
		Evaluation eTestCrossValidation = new Evaluation(trainCrossValidation);
		eTestCrossValidation.crossValidateModel(funSimpleLogistCrossValidation,trainCrossValidation,6,new Random());
		 String strSummaryCrossValidation = eTestCrossValidation.toSummaryString();
		 System.out.println(strSummaryCrossValidation);
		 
	
		File f;
		f = new File(currentPath+File.separator+"Resultados modelo creado con cross validation Arff1.txt");
		FileWriter w = new FileWriter(f);
		BufferedWriter bw = new BufferedWriter(w);	
		PrintWriter wr = new PrintWriter(bw);  
		wr.write(strSummaryCrossValidation);
		wr.close();
		bw.close();
		
		
		
		/*
		//Ejemplos de Classification!
		Classifier smo = new SMO(); //Ojo!! Se puede parametrizar!
		buildTrainTestClassificationModel(currentPath, multiple_instances2,multiple_instances2, smo);
		
		//InputMappedClassifier
		Classifier inputMapped = new InputMappedClassifier();
		Classifier filteredClassifier = new FilteredClassifier();
		((FilteredClassifier)filteredClassifier).setFilter(stwv2);
		((FilteredClassifier)filteredClassifier).setClassifier(new SMO());
		((InputMappedClassifier)inputMapped).setClassifier(filteredClassifier);
		buildTrainTestClassificationModel(currentPath, stwv_instances2,instances2, inputMapped);
		
		*/
		
		return instances2;
	}
	
	public static void main(String[] args) throws Exception{
		if(args.length>0){					
			final String currentPath=args[0];	
			LeerArchivoStopWords();
			stwv_instances1 = GenerarArffPrincipal(currentPath);
			stwv_instances2 = GenerarArff2(currentPath);
			
			
		}
		else{
			System.out.println( "No se paso ninguna ruta");
		//String currentPath="C:\\Cosas\\Quinto 5to  1er cuatri\\mineria nuevo\\phemernrdataset\\pheme-rnr-dataset\\charliehebdo";
	
		//String currentPath =	"C:\\Users\\Yamil\\workspace\\RespaladoMineria\\Data";
		//LeerArchivoStopWords();
		//stwv_instances2 = GenerarArff2(currentPath);
		//stwv_instances1 = GenerarArffPrincipal(currentPath);
		
		}
		/*
		String rutaOrigen="C:\\Cosas\\Quinto 5to  1er cuatri\\mineria nuevo\\phemernrdataset\\pheme-rnr-dataset\\charliehebdo";
		File directorio= new File(rutaOrigen);		
		copiarArchivosJson(directorio, rutaOrigen);


	String currentPath =	"C:\\Users\\Yamil\\workspace\\RespaladoMineria\\Data"

		//String currentPath = System.getProperty("user.dir");
		//String currentPath="C:\\Cosas\\Quinto 5to  1er cuatri\\mineria nuevo\\phemernrdataset\\pheme-rnr-dataset\\charliehebdo\\rumours\\552790824971677696\\reactions";
		//final String currentPath= System.getProperty("user.dir")+"\\Data\\";
		//String currentPath="C:\\Cosas\\Quinto 5to  1er cuatri\\mineria nuevo\\phemernrdataset\\pheme-rnr-dataset\\charliehebdo\\rumours";
		
		LeerArchivoStopWords();	
		System.out.println(currentPath);
		
		Instances stwv_instances1=GenerarArffPrincipal(currentPath);
		Instances stwv_instances2=GenerarArff2(currentPath);
		
			*/
		
		}
	


}



