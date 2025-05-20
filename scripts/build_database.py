import csv
from pymongo import MongoClient
import bson

client = MongoClient('mongodb://localhost:27017/')
print(client.list_database_names())y
db = client['fixed-forms']
collection = db['fixed-forms-collection']

with open('../data/fixed_forms.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    data = [row for row in csvreader]
    
    if data:
        collection.insert_many(data)

        with open('../data/database.bson', 'wb') as bsonfile:
            cursor = collection.find({})
            for doc in cursor:
                bsonfile.write(bson.BSON.encode(doc))
