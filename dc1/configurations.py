from dc1.data_preprocessing.under_over_sampling1 import no_sampling, undersampling, oversampling, \
    smote_sampling

# Sampling method
sampling_methods = {'no_sampling': no_sampling, 'undersampling': undersampling,
                    'oversampling': oversampling,
                    'smote_sampling': smote_sampling}
sampling_method = 'oversampling'

# Remove outliers
data_paths = {'no_modification': '', 'outliers_removed': 'preprocessed/remove_outliers', }
data_path = 'no_modification'
