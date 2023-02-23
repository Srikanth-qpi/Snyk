


list_of_curls = [{"datatype":"tab", "dataset": "https://dataprepfiles.s3.amazonaws.com/CardioGoodFitness.csv", "target_device":"cpu", "num_device":1, "data_prepcmd":'impute_mean', "clmn":"Age,Income,Miles"},
                 {"datatype":"tab", "dataset": "https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv", "target_device":"cpu", "num_device":1, "data_prepcmd":"tsne","category_column_name":"Species","num_components":2,"perplexity":40,"dataframe_delimiter":","},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/samudramanthan_regression/wave_height_train.csv","target_device":"cpu","num_device":1,"data_prepcmd":"outlier_isolation_forest_bivariate","clmn":"wave_height,wind_speed"},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/AutoTabular_Datasets/HR_Analytics/train_LZdllcl.csv","target_device":"cpu","num_device":1,"data_prepcmd":"train_test_split_classification","clmn":"is_promoted"},
                 {"datatype":"tab","dataset":"https://dataprepfiles.s3.amazonaws.com/train.csv","target_device":"cpu","num_device":1,"data_prepcmd":"impute_drop_column","clmn":"Age,Embarked"},
                 {"datatype":"tab","dataset":"https://dataprepfiles.s3.amazonaws.com/train.csv","target_device":"cpu","num_device":1,"data_prepcmd":"impute_median","clmn":"Age"},
                 {"datatype":"tab","dataset":"https://dataprepfiles.s3.amazonaws.com/train.csv","target_device":"cpu","num_device":1,"data_prepcmd":"Impute_KNN"},
                 {"datatype":"tab","dataset":"https://dataprepfiles.s3.amazonaws.com/CardioGoodFitness.csv","target_device":"cpu","num_device":1,"data_prepcmd":"RobustScaler","clmn":"Age,Income,Miles"},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv","target_device":"cpu","num_device":1,"data_prepcmd":"chi_sq","clmn":"Species"},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv","target_device":"cpu","num_device":1,"data_prepcmd":"f_classif","clmn":"Species"},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv","target_device":"cpu","num_device":1,"data_prepcmd":"f_regress","clmn":"Species"},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv","target_device":"cpu","num_device":1,"data_prepcmd":"3D scatter_plot","category_column_name":"Species","clmn":"SepalWidthCm,PetalWidthCm,SepalLengthCm","all_column":"false","dataframe_delimiter":","},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv","target_device":"cpu","num_device":1,"data_prepcmd":"3D scatter_plot","category_column_name":"Species","clmn":"","all_column":"true","dataframe_delimiter":","},
                 {"datatype":"tab","dataset":"https://qpiaidataset.s3.amazonaws.com/SVM/Iris_Dataset.csv","target_device":"cpu","num_device":1,"data_prepcmd":"line_plot","category_column_name":"Species","all_column":"true","dataframe_delimiter":","}]



