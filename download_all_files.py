import kagglehub

# Download latest version
path = kagglehub.dataset_download("caspervanengelenburg/modified-swiss-dwellings")

print("Path to dataset files:", path)