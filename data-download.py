import os
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

# User credentials (replace with your actual credentials or environment variables)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COPERNICUS_USER = os.getenv('COPERNICUS_USER')
COPERNICUS_PASSWORD = os.getenv('COPERNICUS_PASSWORD')
print(COPERNICUS_USER)
print('--------------------------------')
# If environment variables are not set, prompt the user or use placeholders
if not COPERNICUS_USER:
    COPERNICUS_USER = 'your_copernicus_username'
    print("Warning: COPERNICUS_USER environment variable not set. Using placeholder.")
if not COPERNICUS_PASSWORD:
    COPERNICUS_PASSWORD = 'your_copernicus_password'
    print("Warning: COPERNICUS_PASSWORD environment variable not set. Using placeholder.")

# List of product identifiers to download
product_ids = [
    'S2A_MSIL2A_20250706T020121_N0511_R060_T52SFB_20250706T055501'
  
]

# API endpoint for Copernicus Data Space Ecosystem
api_url = 'https://apihub.copernicus.eu/apihub'

def download_sentinel_images(username, password, product_ids, download_dir='.'):
    try:
        api = SentinelAPI(username, password, api_url)
        print(f"Connected to Sentinel API at {api_url}")
    except Exception as e:
        print(f"Error connecting to Sentinel API: {e}")
        return

    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    for product_id in product_ids:
        try:
            # Search for the product by its identifier (Name)
            print(f'Searching for product {product_id}')
            products = api.query(
                productName=f'*{product_id}*',
                platformname='Sentinel-2',
                date=(date(2025, 6, 1), date(2025, 7, 20)), # Broad date range to ensure finding the products
                area=None # No specific area filter needed as we are using product ID
            )

            if not products:
                print(f"Product {product_id} not found.")
                continue

            # Assuming the first result is the correct one
            product_uuid = list(products.keys())[0]
            print(f"Found product {product_id} with UUID {product_uuid}. Downloading...")

            api.download(product_uuid, directory_path=download_dir)
            print(f"Successfully downloaded {product_id}")

        except Exception as e:
            print(f"Error downloading {product_id}: {e}")

if __name__ == "__main__":
    # You should set these as environment variables before running the script
    # For example: export COPERNICUS_USER='your_username' and export COPERNICUS_PASSWORD='your_password'
    # Or, you can directly assign them here for testing, but it's not recommended for production
    
    # Ensure the download directory exists
    download_path = '/Users/jesusjimenez/Documents/MyGitProjects/Super-Resolution of Sentinel-2 Low Resolution'
    os.makedirs(download_path, exist_ok=True)

    download_sentinel_images(COPERNICUS_USER, COPERNICUS_PASSWORD, product_ids, download_path)