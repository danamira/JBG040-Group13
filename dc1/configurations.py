from dc1.data_preprocessing.under_over_sampling1 import no_sampling, undersampling, oversampling, \
    resample_all, smote_sampling

# Sampling method
sampling_methods = {'no_sampling': no_sampling, 'undersampling': undersampling,
                    'oversampling': oversampling, 'resample_all': resample_all,
                    'smote_sampling': smote_sampling}
sampling_method = 'resample_all'

# Remove outliers
data_paths = {'no_modification': '', 'outliers_removed': 'preprocessed/remove_outliers', }
data_path = 'no_modification'
