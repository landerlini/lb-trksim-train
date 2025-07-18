import snakemake_storage_helper 

include: snakemake_storage_helper.__file__

storage s3data:
    **S3StorageProvider.from_secrets('s3data', config.get('secrets_file'))

storage s3models:
    **S3StorageProvider.from_secrets('s3models', config.get('secrets_file'))

storage s3reports:
    **S3StorageProvider.from_secrets('s3reports', config.get('secrets_file'))

storage s3images:
    **S3StorageProvider.from_secrets('s3images', config.get('secrets_file'))

    
