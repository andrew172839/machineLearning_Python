import pandas as pd
train = pd.read_json("/mnt/sda/home/yushao/rental_Listing_Inquiries/train.json")
print train.sample(1)
