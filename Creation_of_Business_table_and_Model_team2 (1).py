# Databricks notebook source
# MAGIC %md
# MAGIC MBD 2021-2022 BIG DATA TOOLS
# MAGIC - Mohammad Hadi Alipour Motlagh
# MAGIC - Sai Sumanth Sripada
# MAGIC - Noor Ayesha

# COMMAND ----------

#PATH OF FILES
PATH_BISINESS="/FileStore/tables/Group Project/parsed_business.json"
PATH_CHECKIN="/FileStore/tables/Group Project/parsed_checkin.json"
PATH_COVID="/FileStore/tables/Group Project/parsed_covid.json"
PATH_REVIEW="/FileStore/tables/Group Project/parsed_review.json"
PATH_TIP="/FileStore/tables/Group Project/parsed_tip.json"
PATH_USER="/FileStore/tables/Group Project/parsed_user.json"

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
import json
import ast
import pyspark
from pyspark.sql.functions import when, lit
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import *
from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

parsed_business_df = spark.read.json(PATH_BISINESS)
parsed_checkin_df  = spark.read.json(PATH_CHECKIN)
parsed_covid_df    = spark.read.json(PATH_COVID)
parsed_review_df   = spark.read.json(PATH_REVIEW)
parsed_tip_df      = spark.read.json(PATH_TIP)
parsed_user_df     = spark.read.json(PATH_USER)

# COMMAND ----------

# MAGIC %md
# MAGIC #  Business DF 

# COMMAND ----------

#Schema for the Business data 
Schema_main_Business = StructType([
  StructField("business_id", StringType(), True),
  StructField("name", StringType(), True),
  StructField("address", StringType(), True),
  StructField("city", StringType(), True),
  StructField("state", StringType(), True),
  StructField("postal_code", StringType(), True),
  StructField("latitude", FloatType(), True),
  StructField("longitude", FloatType(), True),
  StructField("stars", FloatType(), True),
  StructField("review_count", IntegerType(), True),
  StructField("is_open", IntegerType(), True),
  StructField("categories", StringType(), True),
  StructField("attributes.BikeParking", StringType(), True),
  StructField("attributes.GoodForKids", StringType(), True),
  StructField("attributes.BusinessParking", StringType(), True),
  StructField("attributes.ByAppointmentOnly", StringType(), True),
  StructField("attributes.RestaurantsPriceRange2", StringType(), True),
  StructField("hours.Monday", StringType(), True),
  StructField("hours.Tuesday", StringType(), True),
  StructField("hours.Wednesday", StringType(), True),
  StructField("hours.Thursday", StringType(), True),
  StructField("hours.Friday", StringType(), True),
  StructField("hours.Saturday", StringType(), True),
  StructField("hours.Sunday", StringType(), True),
  StructField("attributes.WiFi", StringType(), True),
  StructField("attributes.RestaurantsAttire", StringType(), True),
  StructField("attributes.RestaurantsTakeOut", StringType(), True),
  StructField("attributes.NoiseLevel", StringType(), True),
  StructField("attributes.RestaurantsReservations", StringType(), True),
  StructField("attributes.RestaurantsGoodForGroups", StringType(), True),
  StructField("attributes.HasTV", StringType(), True),
  StructField("attributes.Alcohol", StringType(), True),
  StructField("attributes.RestaurantsDelivery", StringType(), True),
  StructField("attributes.OutdoorSeating", StringType(), True),
  StructField("attributes.Caters", StringType(), True),
  StructField("attributes.Ambience", StringType(), True),
  StructField("attributes.RestaurantsTableService", StringType(), True),
  StructField("attributes.GoodForMeal", StringType(), True),
  StructField("attributes.BusinessAcceptsCreditCards", StringType(), True),
  StructField("attributes.WheelchairAccessible", StringType(), True),
  StructField("attributes.BusinessAcceptsBitcoin", StringType(), True),
  StructField("attributes.DogsAllowed", StringType(), True),
  StructField("attributes.HappyHour", StringType(), True),
  StructField("attributes.GoodForDancing", StringType(), True),
  StructField("attributes.CoatCheck", StringType(), True),
  StructField("attributes.BestNights", StringType(), True),
  StructField("attributes.Music", StringType(), True),
  StructField("attributes.Smoking", StringType(), True),
  StructField("attributes.DriveThru", StringType(), True),
  StructField("attributes.AcceptsInsurance", StringType(), True),
  StructField("attributes.BYOBCorkage", StringType(), True),
  StructField("attributes.HairSpecializesIn", StringType(), True),
  StructField("attributes.Corkage", StringType(), True),
  StructField("attributes.AgesAllowed", StringType(), True),
  StructField("attributes.BYOB", StringType(), True),
  StructField("attributes.DietaryRestrictions", StringType(), True),
  StructField("attributes.RestaurantsCounterService", StringType(), True),
  StructField("attributes.Open24Hours", StringType(), True),
])

# COMMAND ----------

Business = spark\
.read\
.format("json")\
.option("header","true")\
.schema(Schema_main_Business)\
.load(PATH_BISINESS)
Business.display()

# COMMAND ----------

Business.printSchema()

# COMMAND ----------

#Correct the name of the columns
for column in Business.columns:
    if 'attributes' in column:
        Business = Business.withColumnRenamed(column,column.replace('attributes.', ''))

# COMMAND ----------

#Correct the name of the columns
for column in Business.columns:
    if 'hours' in column:
        Business = Business.withColumnRenamed(column,column.replace('hours.', 'H_'))

# COMMAND ----------

Business.display()

# COMMAND ----------

BPS = StructType([
          StructField("garage", BooleanType(), True),
          StructField("street", BooleanType(), True),
          StructField("validated", BooleanType(), True),
          StructField("lot", BooleanType(), True),
          StructField("valet", BooleanType(), True)
    ])
AMBS = StructType([
          StructField("touristy", BooleanType(), True),
          StructField("hipster", BooleanType(), True),
          StructField("romantic", BooleanType(), True),
          StructField("intimate", BooleanType(), True),
          StructField("trendy", BooleanType(), True),
          StructField("upscale", BooleanType(), True),
          StructField("classy", BooleanType(), True),
          StructField("casual", BooleanType(), True),
    ])
GFMS = StructType([
          StructField("dessert", BooleanType(), True),
          StructField("latenight", BooleanType(), True),
          StructField("lunch", BooleanType(), True),
          StructField("dinner", BooleanType(), True),
          StructField("brunch", BooleanType(), True),
          StructField("breakfast", BooleanType(), True)
    ])

BNS = StructType([
          StructField("monday", BooleanType(), True),
          StructField("tuesday", BooleanType(), True),
          StructField("wednesday", BooleanType(), True),
          StructField("thursday", BooleanType(), True),
          StructField("friday", BooleanType(), True),
          StructField("saturday", BooleanType(), True),
          StructField("sunday", BooleanType(), True),
    ])

MUS = StructType([
          StructField("dj", BooleanType(), True),
          StructField("background_music", BooleanType(), True),
          StructField("no_music", BooleanType(), True),
          StructField("jukebox", BooleanType(), True),
          StructField("live", BooleanType(), True),
          StructField("video", BooleanType(), True),
          StructField("karaoke", BooleanType(), True),
    ])

HSS = StructType([
          StructField("straightperms", BooleanType(), True),
          StructField("coloring", BooleanType(), True),
          StructField("extensions", BooleanType(), True),
          StructField("africanamerican", BooleanType(), True),
          StructField("curly", BooleanType(), True),
          StructField("kids", BooleanType(), True),
          StructField("perms", BooleanType(), True),
          StructField("asian", BooleanType(), True),
    ])
dict_to_json = udf(lambda x: json.dumps(ast.literal_eval(str(x))))

Business_REP = Business\
.withColumn("BusinessParking", from_json(dict_to_json("BusinessParking"),BPS))\
.withColumn("Ambience", from_json(dict_to_json("Ambience"),AMBS))\
.withColumn("GoodForMeal", from_json(dict_to_json("GoodForMeal"),GFMS))\
.withColumn("BestNights", from_json(dict_to_json("BestNights"),BNS))\
.withColumn("Music", from_json(dict_to_json("Music"),MUS))\
.withColumn("HairSpecializesIn", from_json(dict_to_json("HairSpecializesIn"),HSS))\
.select("*","BusinessParking.*","Ambience.*","GoodForMeal.*","BestNights.*","Music.*","HairSpecializesIn.*")\
.drop(col("BusinessParking"))\
.drop(col("Ambience"))\
.drop(col("GoodForMeal"))\
.drop(col("BestNights"))\
.drop(col("Music"))\
.drop(col("HairSpecializesIn"))
Business_REP.display()

# COMMAND ----------

Business_REP = Business_REP\
.withColumn("Active Life", when(Business_REP.categories.contains("Amateur Sports Teams"),lit(1))\
.when(Business_REP.categories.contains("Amusement Parks"),lit(1))\
.when(Business_REP.categories.contains("Aquariums"),lit(1))\
.when(Business_REP.categories.contains("Archery"),lit(1))\
.when(Business_REP.categories.contains("Badminton"),lit(1))\
.when(Business_REP.categories.contains("Basketball Courts"),lit(1))\
.when(Business_REP.categories.contains("Beaches"),lit(1))\
.when(Business_REP.categories.contains("Bike Rentals"),lit(1))\
.when(Business_REP.categories.contains("Boating"),lit(1))\
.when(Business_REP.categories.contains("Bowling"),lit(1))\
.when(Business_REP.categories.contains("Climbing"),lit(1))\
.when(Business_REP.categories.contains("Disc Golf"),lit(1))\
.when(Business_REP.categories.contains("Diving"),lit(1))\
.when(Business_REP.categories.contains("Free Diving"),lit(1))\
.when(Business_REP.categories.contains("Scuba Diving"),lit(1))\
.when(Business_REP.categories.contains("Fishing"),lit(1))\
.when(Business_REP.categories.contains("Fitness & Instruction"),lit(1))\
.when(Business_REP.categories.contains("Barre Classes"),lit(1))\
.when(Business_REP.categories.contains("Boot Camps"),lit(1))\
.when(Business_REP.categories.contains("Boxing"),lit(1))\
.when(Business_REP.categories.contains("Dance Studios"),lit(1))\
.when(Business_REP.categories.contains("Gyms"),lit(1))\
.when(Business_REP.categories.contains("Martial Arts"),lit(1))\
.when(Business_REP.categories.contains("Pilates"),lit(1))\
.when(Business_REP.categories.contains("Swimming Lessons/Schools"),lit(1))\
.when(Business_REP.categories.contains("Tai Chi"),lit(1))\
.when(Business_REP.categories.contains("Trainers"),lit(1))\
.when(Business_REP.categories.contains("Yoga"),lit(1))\
.when(Business_REP.categories.contains("Go Karts"),lit(1))\
.when(Business_REP.categories.contains("Gun/Rifle Ranges"),lit(1))\
.when(Business_REP.categories.contains("Gymnastics"),lit(1))\
.when(Business_REP.categories.contains("Hang Gliding"),lit(1))\
.when(Business_REP.categories.contains("Hiking"),lit(1))\
.when(Business_REP.categories.contains("Horse Racing"),lit(1))\
.when(Business_REP.categories.contains("Horseback Riding"),lit(1))\
.when(Business_REP.categories.contains("Hot Air Balloons"),lit(1))\
.when(Business_REP.categories.contains("Kiteboarding"),lit(1))\
.when(Business_REP.categories.contains("Lakes"),lit(1))\
.when(Business_REP.categories.contains("Laser Tag"),lit(1))\
.when(Business_REP.categories.contains("Leisure Centers"),lit(1))\
.when(Business_REP.categories.contains("Mini Golf"),lit(1))\
.when(Business_REP.categories.contains("Mountain Biking"),lit(1))\
.when(Business_REP.categories.contains("Paddleboarding"),lit(1))\
.when(Business_REP.categories.contains("Paintball"),lit(1))\
.when(Business_REP.categories.contains("Parks"),lit(1))\
.when(Business_REP.categories.contains("Dog Parks"),lit(1))\
.when(Business_REP.categories.contains("Skate Parks"),lit(1))\
.when(Business_REP.categories.contains("Playgrounds"),lit(1))\
.when(Business_REP.categories.contains("Rafting/Kayaking"),lit(1))\
.when(Business_REP.categories.contains("Recreation Centers"),lit(1))\
.when(Business_REP.categories.contains("Rock Climbing"),lit(1))\
.when(Business_REP.categories.contains("Skating Rinks"),lit(1))\
.when(Business_REP.categories.contains("Skydiving"),lit(1))\
.when(Business_REP.categories.contains("Soccer"),lit(1))\
.when(Business_REP.categories.contains("Spin Classes"),lit(1))\
.when(Business_REP.categories.contains("Sports Clubs"),lit(1))\
.when(Business_REP.categories.contains("Squash"),lit(1))\
.when(Business_REP.categories.contains("Summer Camps"),lit(1))\
.when(Business_REP.categories.contains("Surfing"),lit(1))\
.when(Business_REP.categories.contains("Swimming Pools"),lit(1))\
.when(Business_REP.categories.contains("Tennis"),lit(1))\
.when(Business_REP.categories.contains("Trampoline Parks"),lit(1))\
.when(Business_REP.categories.contains("Tubing"),lit(1))\
.when(Business_REP.categories.contains("Zoos"),lit(1))\
.otherwise(0))


# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Arts & Entertainment", when(Business_REP.categories.contains('Arcades'),lit(1))\
.when(Business_REP.categories.contains('Art Galleries'),lit(1))\
.when(Business_REP.categories.contains('Botanical Gardens'),lit(1))\
.when(Business_REP.categories.contains('Casinos'),lit(1))\
.when(Business_REP.categories.contains('Cinema'),lit(1))\
.when(Business_REP.categories.contains('Cultural Center'),lit(1))\
.when(Business_REP.categories.contains('Festivals'),lit(1))\
.when(Business_REP.categories.contains('Jazz & Blues'),lit(1))\
.when(Business_REP.categories.contains('Museums'),lit(1))\
.when(Business_REP.categories.contains('Music Venues'),lit(1))\
.when(Business_REP.categories.contains('Opera & Ballet'),lit(1))\
.when(Business_REP.categories.contains('Performing Arts'),lit(1))\
.when(Business_REP.categories.contains('Professional Sports Teams'),lit(1))\
.when(Business_REP.categories.contains('Psychics & Astrologers'),lit(1))\
.when(Business_REP.categories.contains('Race Tracks'),lit(1))\
.when(Business_REP.categories.contains('Social Clubs'),lit(1))\
.when(Business_REP.categories.contains('Stadiums & Arenas'),lit(1))\
.when(Business_REP.categories.contains('Ticket Sales'),lit(1))\
.when(Business_REP.categories.contains('Wineries'),lit(1))\
.otherwise(0))


# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Automotive", when(Business_REP.categories.contains('Auto Detailing'),lit(1))\
.when(Business_REP.categories.contains('Auto Glass Services'),lit(1))\
.when(Business_REP.categories.contains('Auto Loan Providers'),lit(1))\
.when(Business_REP.categories.contains('Auto Parts & Supplies'),lit(1))\
.when(Business_REP.categories.contains('Auto Repair'),lit(1))\
.when(Business_REP.categories.contains('Boat Dealers'),lit(1))\
.when(Business_REP.categories.contains('Body Shops'),lit(1))\
.when(Business_REP.categories.contains('Car Dealers'),lit(1))\
.when(Business_REP.categories.contains('Car Stereo Installation'),lit(1))\
.when(Business_REP.categories.contains('Car Wash'),lit(1))\
.when(Business_REP.categories.contains('Gas & Service Stations'),lit(1))\
.when(Business_REP.categories.contains('Motorcycle Dealers'),lit(1))\
.when(Business_REP.categories.contains('Motorcycle Repair'),lit(1))\
.when(Business_REP.categories.contains('Oil Change Stations'),lit(1))\
.when(Business_REP.categories.contains('Parking'),lit(1))\
.when(Business_REP.categories.contains('RV Dealers'),lit(1))\
.when(Business_REP.categories.contains('Smog Check Stations'),lit(1))\
.when(Business_REP.categories.contains('Tires'),lit(1))\
.when(Business_REP.categories.contains('Towing'),lit(1))\
.when(Business_REP.categories.contains('Truck Rental'),lit(1))\
.when(Business_REP.categories.contains('Windshield Installation & Repair'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Beauty & Spas", when(Business_REP.categories.contains('Barbers'),lit(1))\
.when(Business_REP.categories.contains('Cosmetics & Beauty Supply'),lit(1))\
.when(Business_REP.categories.contains('Day Spas'),lit(1))\
.when(Business_REP.categories.contains('Eyelash Service'),lit(1))\
.when(Business_REP.categories.contains('Hair Extensions'),lit(1))\
.when(Business_REP.categories.contains('Hair Removal'),lit(1))\
.when(Business_REP.categories.contains('Laser Hair Removal'),lit(1))\
.when(Business_REP.categories.contains('Hair Salons'),lit(1))\
.when(Business_REP.categories.contains('Blow Dry/Out Services'),lit(1))\
.when(Business_REP.categories.contains('Hair Extensions'),lit(1))\
.when(Business_REP.categories.contains('Hair Stylists'),lit(1))\
.when(Business_REP.categories.contains("Men's Hair Salons"),lit(1))\
.when(Business_REP.categories.contains('Makeup Artists'),lit(1))\
.when(Business_REP.categories.contains('Massage'),lit(1))\
.when(Business_REP.categories.contains('Medical Spas'),lit(1))\
.when(Business_REP.categories.contains('Nail Salons'),lit(1))\
.when(Business_REP.categories.contains('Permanent Makeup'),lit(1))\
.when(Business_REP.categories.contains('Piercing'),lit(1))\
.when(Business_REP.categories.contains('Rolfing'),lit(1))\
.when(Business_REP.categories.contains('Skin care'),lit(1))\
.when(Business_REP.categories.contains('Tanning'),lit(1))\
.when(Business_REP.categories.contains('Spray Tanning'),lit(1))\
.when(Business_REP.categories.contains('Tanning Beds'),lit(1))\
.when(Business_REP.categories.contains('Tattoo'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Education", when(Business_REP.categories.contains('Adult Education'),lit(1))\
.when(Business_REP.categories.contains('College Counseling'),lit(1))\
.when(Business_REP.categories.contains('Colleges & Universities'),lit(1))\
.when(Business_REP.categories.contains('Educational Services'),lit(1))\
.when(Business_REP.categories.contains('Elementary Schools'),lit(1))\
.when(Business_REP.categories.contains('Middle Schools & High Schools'),lit(1))\
.when(Business_REP.categories.contains('Preschools'),lit(1))\
.when(Business_REP.categories.contains('Private Tutors'),lit(1))\
.when(Business_REP.categories.contains('Religious Schools'),lit(1))\
.when(Business_REP.categories.contains('Special Education'),lit(1))\
.when(Business_REP.categories.contains('Specialty Schools'),lit(1))\
.when(Business_REP.categories.contains('Art Schools'),lit(1))\
.when(Business_REP.categories.contains('CPR Classes'),lit(1))\
.when(Business_REP.categories.contains('Cooking Schools'),lit(1))\
.when(Business_REP.categories.contains('Cosmetology Schools'),lit(1))\
.when(Business_REP.categories.contains('Dance Schools'),lit(1))\
.when(Business_REP.categories.contains('Driving Schools'),lit(1))\
.when(Business_REP.categories.contains('First Aid Classes'),lit(1))\
.when(Business_REP.categories.contains('Flight Instruction'),lit(1))\
.when(Business_REP.categories.contains('Language Schools'),lit(1))\
.when(Business_REP.categories.contains('Massage Schools'),lit(1))\
.when(Business_REP.categories.contains('Swimming Lessons/Schools'),lit(1))\
.when(Business_REP.categories.contains('Vocational & Technical School'),lit(1))\
.when(Business_REP.categories.contains('Test Preparation'),lit(1))\
.when(Business_REP.categories.contains('Tutoring Centers'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Event Planning & Services", when(Business_REP.categories.contains('Bartenders'),lit(1))\
.when(Business_REP.categories.contains('Boat Charters'),lit(1))\
.when(Business_REP.categories.contains('Cards & Stationery'),lit(1))\
.when(Business_REP.categories.contains('Caterers'),lit(1))\
.when(Business_REP.categories.contains('Clowns'),lit(1))\
.when(Business_REP.categories.contains('DJs'),lit(1))\
.when(Business_REP.categories.contains('Hotels'),lit(1))\
.when(Business_REP.categories.contains('Magicians'),lit(1))\
.when(Business_REP.categories.contains('Musicians'),lit(1))\
.when(Business_REP.categories.contains('Officiants'),lit(1))\
.when(Business_REP.categories.contains('Part & Event Planning'),lit(1))\
.when(Business_REP.categories.contains('Party Bus Rentals'),lit(1))\
.when(Business_REP.categories.contains('Party Equipment Rentals'),lit(1))\
.when(Business_REP.categories.contains('Party Supplies'),lit(1))\
.when(Business_REP.categories.contains('Personal Chefs'),lit(1))\
.when(Business_REP.categories.contains('Photographers'),lit(1))\
.when(Business_REP.categories.contains('Event Photography'),lit(1))\
.when(Business_REP.categories.contains('Session Photography'),lit(1))\
.when(Business_REP.categories.contains('Venues & Event Spaces'),lit(1))\
.when(Business_REP.categories.contains('Videographers'),lit(1))\
.when(Business_REP.categories.contains('Wedding Planning'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Financial Services", when(Business_REP.categories.contains('Banks & Credit Unions'),lit(1))\
.when(Business_REP.categories.contains('Check Cashing/Pay-day Loans'),lit(1))\
.when(Business_REP.categories.contains('Financial Advising'),lit(1))\
.when(Business_REP.categories.contains('Insurance'),lit(1))\
.when(Business_REP.categories.contains('Investing'),lit(1))\
.when(Business_REP.categories.contains('Tax Services'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Food", when(Business_REP.categories.contains('Bagels'),lit(1))\
.when(Business_REP.categories.contains('Bakeries'),lit(1))\
.when(Business_REP.categories.contains('Beer, Wine & Spirits'),lit(1))\
.when(Business_REP.categories.contains('Breweries'),lit(1))\
.when(Business_REP.categories.contains('Bubble Tea'),lit(1))\
.when(Business_REP.categories.contains('Butcher'),lit(1))\
.when(Business_REP.categories.contains('CSA'),lit(1))\
.when(Business_REP.categories.contains('Coffee & Tea'),lit(1))\
.when(Business_REP.categories.contains('Convenience Stores'),lit(1))\
.when(Business_REP.categories.contains('Desserts'),lit(1))\
.when(Business_REP.categories.contains('Do-It-Yourself Food'),lit(1))\
.when(Business_REP.categories.contains('Donuts'),lit(1))\
.when(Business_REP.categories.contains('Farmers Market'),lit(1))\
.when(Business_REP.categories.contains('Food Delivery Services'),lit(1))\
.when(Business_REP.categories.contains('Food Trucks'),lit(1))\
.when(Business_REP.categories.contains('Gelato'),lit(1))\
.when(Business_REP.categories.contains('Grocery'),lit(1))\
.when(Business_REP.categories.contains('Ice Cream & Frozen Yogust'),lit(1))\
.when(Business_REP.categories.contains('Internet Cafes'),lit(1))\
.when(Business_REP.categories.contains('Juice Bars & Smoothies'),lit(1))\
.when(Business_REP.categories.contains('Pretzels'),lit(1))\
.when(Business_REP.categories.contains('Shaved Ice'),lit(1))\
.when(Business_REP.categories.contains('Specialty Food'),lit(1))\
.when(Business_REP.categories.contains('Candy Stores'),lit(1))\
.when(Business_REP.categories.contains('Cheese Shops'),lit(1))\
.when(Business_REP.categories.contains('Chocolatiers & Shops'),lit(1))\
.when(Business_REP.categories.contains('Ethnic Food'),lit(1))\
.when(Business_REP.categories.contains('Fruits & Veggies'),lit(1))\
.when(Business_REP.categories.contains('Health Markets'),lit(1))\
.when(Business_REP.categories.contains('Herbs & Spices'),lit(1))\
.when(Business_REP.categories.contains('Meat Shops'),lit(1))\
.when(Business_REP.categories.contains('Seafood Markets'),lit(1))\
.when(Business_REP.categories.contains('Street Vendors'),lit(1))\
.when(Business_REP.categories.contains('Tea Rooms'),lit(1))\
.when(Business_REP.categories.contains('Wineries'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Columns with categories
Business_REP = Business_REP\
.withColumn("Health & Medical", when(Business_REP.categories.contains('Acupuncture'),lit(1))\
.when(Business_REP.categories.contains('Cannabis Clinics'),lit(1))\
.when(Business_REP.categories.contains('Chiropractors'),lit(1))\
.when(Business_REP.categories.contains('Counseling & Mental Health'),lit(1))\
.when(Business_REP.categories.contains('Dentists'),lit(1))\
.when(Business_REP.categories.contains('Cosmetic Dentists'),lit(1))\
.when(Business_REP.categories.contains('Endodontists'),lit(1))\
.when(Business_REP.categories.contains('General Dentistry'),lit(1))\
.when(Business_REP.categories.contains('Oral Surgeons'),lit(1))\
.when(Business_REP.categories.contains('Orthodontists'),lit(1))\
.when(Business_REP.categories.contains('Pediatric Dentists'),lit(1))\
.when(Business_REP.categories.contains('Periodontists'),lit(1))\
.when(Business_REP.categories.contains('Diagnostic Services'),lit(1))\
.when(Business_REP.categories.contains('Diagnostic Imaging'),lit(1))\
.when(Business_REP.categories.contains('Laboratory Testing'),lit(1))\
.when(Business_REP.categories.contains('Doctors'),lit(1))\
.when(Business_REP.categories.contains('Allergists'),lit(1))\
.when(Business_REP.categories.contains('Anesthesiologists'),lit(1))\
.when(Business_REP.categories.contains('Audiologist'),lit(1))\
.when(Business_REP.categories.contains('Cardiologists'),lit(1))\
.when(Business_REP.categories.contains('Cosmetic Surgeons'),lit(1))\
.when(Business_REP.categories.contains('Dermatologists'),lit(1))\
.when(Business_REP.categories.contains('Ear Nose & Throat'),lit(1))\
.when(Business_REP.categories.contains('Family Practice'),lit(1))\
.when(Business_REP.categories.contains('Fertility'),lit(1))\
.when(Business_REP.categories.contains('Gastroenterologist'),lit(1))\
.when(Business_REP.categories.contains('Gerontologists'),lit(1))\
.when(Business_REP.categories.contains('Internal Medicine'),lit(1))\
.when(Business_REP.categories.contains('Naturopathic/Holistic'),lit(1))\
.when(Business_REP.categories.contains('Neurologist'),lit(1))\
.when(Business_REP.categories.contains('Obstretricians & Gynecologists'),lit(1))\
.when(Business_REP.categories.contains('Oncologist'),lit(1))\
.when(Business_REP.categories.contains('Ophthalmologists'),lit(1))\
.when(Business_REP.categories.contains('Orthopedists'),lit(1))\
.when(Business_REP.categories.contains('Osteopathic Physicians'),lit(1))\
.when(Business_REP.categories.contains('Pediatricians'),lit(1))\
.when(Business_REP.categories.contains('Podiatrists'),lit(1))\
.when(Business_REP.categories.contains('Proctologists'),lit(1))\
.when(Business_REP.categories.contains('Psychiatrists'),lit(1))\
.when(Business_REP.categories.contains('Pulmonologists'),lit(1))\
.when(Business_REP.categories.contains('Sports Medicine'),lit(1))\
.when(Business_REP.categories.contains('Tattoo Removal'),lit(1))\
.when(Business_REP.categories.contains('Urologists'),lit(1))\
.when(Business_REP.categories.contains('Hearing Aid Providers'),lit(1))\
.when(Business_REP.categories.contains('Home Health Care'),lit(1))\
.when(Business_REP.categories.contains('Hospice'),lit(1))\
.when(Business_REP.categories.contains('Hospitals'),lit(1))\
.when(Business_REP.categories.contains('Lactation Services'),lit(1))\
.when(Business_REP.categories.contains('Laser Eye Surgery/Lasik'),lit(1))\
.when(Business_REP.categories.contains('Massage Therapy'),lit(1))\
.when(Business_REP.categories.contains('Medical Centers'),lit(1))\
.when(Business_REP.categories.contains('Medical Spas'),lit(1))\
.when(Business_REP.categories.contains('Medical Transportation'),lit(1))\
.when(Business_REP.categories.contains('Midwives'),lit(1))\
.when(Business_REP.categories.contains('Nutritionists'),lit(1))\
.when(Business_REP.categories.contains('Occupational Therapy'),lit(1))\
.when(Business_REP.categories.contains('Optometrists'),lit(1))\
.when(Business_REP.categories.contains('Physical Therapy'),lit(1))\
.when(Business_REP.categories.contains('Reflexology'),lit(1))\
.when(Business_REP.categories.contains('Rehabilitation Center'),lit(1))\
.when(Business_REP.categories.contains('Retirement Homes'),lit(1))\
.when(Business_REP.categories.contains('Speech Therapists'),lit(1))\
.when(Business_REP.categories.contains('Traditional Chinese Medicine'),lit(1))\
.when(Business_REP.categories.contains('Urgent Care'),lit(1))\
.when(Business_REP.categories.contains('Weight Loss Centers'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Home Services
Business_REP = Business_REP\
.withColumn("Home Services", when(Business_REP.categories.contains('Building Supplies'),lit(1))\
.when(Business_REP.categories.contains('Carpet Installation'),lit(1))\
.when(Business_REP.categories.contains('Carpeting'),lit(1))\
.when(Business_REP.categories.contains('Contractors'),lit(1))\
.when(Business_REP.categories.contains('Damage Restoration'),lit(1))\
.when(Business_REP.categories.contains('Electricians'),lit(1))\
.when(Business_REP.categories.contains('Flooring'),lit(1))\
.when(Business_REP.categories.contains('Garage Door Services'),lit(1))\
.when(Business_REP.categories.contains('Gardeners'),lit(1))\
.when(Business_REP.categories.contains('Handyman'),lit(1))\
.when(Business_REP.categories.contains('Heating & Air Conditioning/HVAC'),lit(1))\
.when(Business_REP.categories.contains('Home Cleaning'),lit(1))\
.when(Business_REP.categories.contains('Home Inspectors'),lit(1))\
.when(Business_REP.categories.contains('Home Organization'),lit(1))\
.when(Business_REP.categories.contains('Home Theatre Installation'),lit(1))\
.when(Business_REP.categories.contains('Home Window Tinting'),lit(1))\
.when(Business_REP.categories.contains('Interior Design'),lit(1))\
.when(Business_REP.categories.contains('Internet Service Providers'),lit(1))\
.when(Business_REP.categories.contains('Irrigation'),lit(1))\
.when(Business_REP.categories.contains('Keys & Locksmith'),lit(1))\
.when(Business_REP.categories.contains('Landscape Architects'),lit(1))\
.when(Business_REP.categories.contains('Landscaping'),lit(1))\
.when(Business_REP.categories.contains('Lighting Fixtures & Equipment'),lit(1))\
.when(Business_REP.categories.contains('Masonry/Concrete'),lit(1))\
.when(Business_REP.categories.contains('Movers'),lit(1))\
.when(Business_REP.categories.contains('Painters'),lit(1))\
.when(Business_REP.categories.contains('Plumbing'),lit(1))\
.when(Business_REP.categories.contains('Pool Cleaners'),lit(1))\
.when(Business_REP.categories.contains('Real Estate'),lit(1))\
.when(Business_REP.categories.contains('Apartments'),lit(1))\
.when(Business_REP.categories.contains('Commercial Real Estate'),lit(1))\
.when(Business_REP.categories.contains('Home Staging'),lit(1))\
.when(Business_REP.categories.contains('Mortgage Brokers'),lit(1))\
.when(Business_REP.categories.contains('Property Management'),lit(1))\
.when(Business_REP.categories.contains('Real Estate Agents'),lit(1))\
.when(Business_REP.categories.contains('Real Estate Services'),lit(1))\
.when(Business_REP.categories.contains('Shared Office Spaces'),lit(1))\
.when(Business_REP.categories.contains('University Housing'),lit(1))\
.when(Business_REP.categories.contains('Roofing'),lit(1))\
.when(Business_REP.categories.contains('Security Systems'),lit(1))\
.when(Business_REP.categories.contains('Shades & Blinds'),lit(1))\
.when(Business_REP.categories.contains('Solar Installation'),lit(1))\
.when(Business_REP.categories.contains('Television Service Providers'),lit(1))\
.when(Business_REP.categories.contains('Tree Services'),lit(1))\
.when(Business_REP.categories.contains('Utilities'),lit(1))\
.when(Business_REP.categories.contains('Window Washing'),lit(1))\
.when(Business_REP.categories.contains('Windows Installation'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Hotels & Travel
Business_REP = Business_REP\
.withColumn("Hotels & Travel", when(Business_REP.categories.contains('Airports'),lit(1))\
.when(Business_REP.categories.contains('Bed & Breakfast'),lit(1))\
.when(Business_REP.categories.contains('Campgrounds'),lit(1))\
.when(Business_REP.categories.contains('Car Rental'),lit(1))\
.when(Business_REP.categories.contains('Guest Houses'),lit(1))\
.when(Business_REP.categories.contains('Hostels'),lit(1))\
.when(Business_REP.categories.contains('Hotels'),lit(1))\
.when(Business_REP.categories.contains('Motorcycle Rental'),lit(1))\
.when(Business_REP.categories.contains('RV Parks'),lit(1))\
.when(Business_REP.categories.contains('RV Rental'),lit(1))\
.when(Business_REP.categories.contains('Resorts'),lit(1))\
.when(Business_REP.categories.contains('Ski Resorts'),lit(1))\
.when(Business_REP.categories.contains('Tours'),lit(1))\
.when(Business_REP.categories.contains('Train Stations'),lit(1))\
.when(Business_REP.categories.contains('Transportation'),lit(1))\
.when(Business_REP.categories.contains('Airlines'),lit(1))\
.when(Business_REP.categories.contains('Airport Shuttles'),lit(1))\
.when(Business_REP.categories.contains('Limos'),lit(1))\
.when(Business_REP.categories.contains('Public Transportation'),lit(1))\
.when(Business_REP.categories.contains('Taxis'),lit(1))\
.when(Business_REP.categories.contains('Travel Services'),lit(1))\
.when(Business_REP.categories.contains('Vacation Rental Agents'),lit(1))\
.when(Business_REP.categories.contains('Vacation Rentals'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Local Flavor
Business_REP = Business_REP\
.withColumn("Local Flavor", when(Business_REP.categories.contains('Yelp Events'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Local Services
Business_REP = Business_REP\
.withColumn("Local Services", when(Business_REP.categories.contains('Appliances & Repair'),lit(1))\
.when(Business_REP.categories.contains('Bail Bondsmen'),lit(1))\
.when(Business_REP.categories.contains('Bike Repair/Maintenance'),lit(1))\
.when(Business_REP.categories.contains('Carpet Cleaning'),lit(1))\
.when(Business_REP.categories.contains('Child Care & Day Care'),lit(1))\
.when(Business_REP.categories.contains('Community Service/Non-Profit'),lit(1))\
.when(Business_REP.categories.contains('Couriers & Delivery Services'),lit(1))\
.when(Business_REP.categories.contains('Dry Cleaning & Laundry'),lit(1))\
.when(Business_REP.categories.contains('Electronics Repair'),lit(1))\
.when(Business_REP.categories.contains('Funeral Services & Cemeteries'),lit(1))\
.when(Business_REP.categories.contains('Furniture Reupholstery'),lit(1))\
.when(Business_REP.categories.contains('IT Services & Computer Repair'),lit(1))\
.when(Business_REP.categories.contains('Data Recovery'),lit(1))\
.when(Business_REP.categories.contains('Mobile Phone Repair'),lit(1))\
.when(Business_REP.categories.contains('Jewelry Repair'),lit(1))\
.when(Business_REP.categories.contains('Junk Removal & Hauling'),lit(1))\
.when(Business_REP.categories.contains('Nanny Services'),lit(1))\
.when(Business_REP.categories.contains('Notaries'),lit(1))\
.when(Business_REP.categories.contains('Pest Control'),lit(1))\
.when(Business_REP.categories.contains('Printing Services'),lit(1))\
.when(Business_REP.categories.contains('Recording & Rehearsal Studios'),lit(1))\
.when(Business_REP.categories.contains('Recycling Center'),lit(1))\
.when(Business_REP.categories.contains('Screen Printing'),lit(1))\
.when(Business_REP.categories.contains('Screen Printing/T-Shirt Printing'),lit(1))\
.when(Business_REP.categories.contains('Self Storage'),lit(1))\
.when(Business_REP.categories.contains('Sewing & Alterations'),lit(1))\
.when(Business_REP.categories.contains('Shipping Centers'),lit(1))\
.when(Business_REP.categories.contains('Shoe Repair'),lit(1))\
.when(Business_REP.categories.contains('Snow Removal'),lit(1))\
.when(Business_REP.categories.contains('Watch Repair'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Mass Media
Business_REP = Business_REP\
.withColumn("Mass Media", when(Business_REP.categories.contains('Print Media'),lit(1))\
.when(Business_REP.categories.contains('Radio Stations'),lit(1))\
.when(Business_REP.categories.contains('Television Stations'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Nighlife
Business_REP = Business_REP\
.withColumn("Nightlife", when(Business_REP.categories.contains('Adult Entertainment'),lit(1))\
.when(Business_REP.categories.contains('Bars'),lit(1))\
.when(Business_REP.categories.contains('Champagne Bars'),lit(1))\
.when(Business_REP.categories.contains('Cocktail Bars'),lit(1))\
.when(Business_REP.categories.contains('Dive Bars'),lit(1))\
.when(Business_REP.categories.contains('Gay Bars'),lit(1))\
.when(Business_REP.categories.contains('Hookah Bars'),lit(1))\
.when(Business_REP.categories.contains('Lounges'),lit(1))\
.when(Business_REP.categories.contains('Pubs'),lit(1))\
.when(Business_REP.categories.contains('Sports Bars'),lit(1))\
.when(Business_REP.categories.contains('Wine Bars'),lit(1))\
.when(Business_REP.categories.contains('Comedy Clubs'),lit(1))\
.when(Business_REP.categories.contains('Country Dance Halls'),lit(1))\
.when(Business_REP.categories.contains('Dance Clubs'),lit(1))\
.when(Business_REP.categories.contains('Jazz & Blues'),lit(1))\
.when(Business_REP.categories.contains('Karaoke'),lit(1))\
.when(Business_REP.categories.contains('Music Venues'),lit(1))\
.when(Business_REP.categories.contains('Piano Bars'),lit(1))\
.when(Business_REP.categories.contains('Pool Halls'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Pets
Business_REP = Business_REP\
.withColumn("Pets", when(Business_REP.categories.contains('Animal Shelters'),lit(1))\
.when(Business_REP.categories.contains('Horse Boarding'),lit(1))\
.when(Business_REP.categories.contains('Pet Services'),lit(1))\
.when(Business_REP.categories.contains('Dog Walkers'),lit(1))\
.when(Business_REP.categories.contains('Pet Boarding/Pet Sitting'),lit(1))\
.when(Business_REP.categories.contains('Pet Groomers'),lit(1))\
.when(Business_REP.categories.contains('Pet Training'),lit(1))\
.when(Business_REP.categories.contains('Pet Stores'),lit(1))\
.when(Business_REP.categories.contains('Veterinarians'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Professioanl Services
Business_REP = Business_REP\
.withColumn("Professional Services", when(Business_REP.categories.contains('Accountants'),lit(1))\
.when(Business_REP.categories.contains('Advertising'),lit(1))\
.when(Business_REP.categories.contains('Architects'),lit(1))\
.when(Business_REP.categories.contains('Boat Repair'),lit(1))\
.when(Business_REP.categories.contains('Career Counseling'),lit(1))\
.when(Business_REP.categories.contains('Editorial Services'),lit(1))\
.when(Business_REP.categories.contains('Employment Agencies'),lit(1))\
.when(Business_REP.categories.contains('Graphic Design'),lit(1))\
.when(Business_REP.categories.contains('Internet Service Providers'),lit(1))\
.when(Business_REP.categories.contains('Lawyers'),lit(1))\
.when(Business_REP.categories.contains('Bankruptcy Law'),lit(1))\
.when(Business_REP.categories.contains('Business Law'),lit(1))\
.when(Business_REP.categories.contains('Criminal Defense Law'),lit(1))\
.when(Business_REP.categories.contains('DUI Law'),lit(1))\
.when(Business_REP.categories.contains('Divorce & Family Law'),lit(1))\
.when(Business_REP.categories.contains('Employment Law'),lit(1))\
.when(Business_REP.categories.contains('Estate Planning Law'),lit(1))\
.when(Business_REP.categories.contains('General Litigation'),lit(1))\
.when(Business_REP.categories.contains('Immigration Law'),lit(1))\
.when(Business_REP.categories.contains('Personal Injury Law'),lit(1))\
.when(Business_REP.categories.contains('Real Estate Law'),lit(1))\
.when(Business_REP.categories.contains('Legal Services'),lit(1))\
.when(Business_REP.categories.contains('Life Coach'),lit(1))\
.when(Business_REP.categories.contains('Marketing'),lit(1))\
.when(Business_REP.categories.contains('Matchmakers'),lit(1))\
.when(Business_REP.categories.contains('Office Cleaning'),lit(1))\
.when(Business_REP.categories.contains('Payroll Services'),lit(1))\
.when(Business_REP.categories.contains('Personal Assistants'),lit(1))\
.when(Business_REP.categories.contains('Private Investigation'),lit(1))\
.when(Business_REP.categories.contains('Public Relations'),lit(1))\
.when(Business_REP.categories.contains('Talent Agencies'),lit(1))\
.when(Business_REP.categories.contains('Taxidermy'),lit(1))\
.when(Business_REP.categories.contains('Translation Services'),lit(1))\
.when(Business_REP.categories.contains('Video/Film Production'),lit(1))\
.when(Business_REP.categories.contains('Web Design'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Public Services & Government
Business_REP = Business_REP\
.withColumn("Public Services & Government", when(Business_REP.categories.contains('Courthouses'),lit(1))\
.when(Business_REP.categories.contains('Departments of Motor Vehicles'),lit(1))\
.when(Business_REP.categories.contains('Embassy'),lit(1))\
.when(Business_REP.categories.contains('Fire Departments'),lit(1))\
.when(Business_REP.categories.contains('Landmarks & Historical Buildings'),lit(1))\
.when(Business_REP.categories.contains('Libraries'),lit(1))\
.when(Business_REP.categories.contains('Police Departments'),lit(1))\
.when(Business_REP.categories.contains('Post Offices'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Real Estate
Business_REP = Business_REP\
.withColumn("Real Estate", when(Business_REP.categories.contains('Apartments'),lit(1))\
.when(Business_REP.categories.contains('Commercial Real Estate'),lit(1))\
.when(Business_REP.categories.contains('Home Staging'),lit(1))\
.when(Business_REP.categories.contains('Mortgage Brokers'),lit(1))\
.when(Business_REP.categories.contains('Property Management'),lit(1))\
.when(Business_REP.categories.contains('Real Estate Agents'),lit(1))\
.when(Business_REP.categories.contains('Real Estate Services'),lit(1))\
.when(Business_REP.categories.contains('Shared Office Spaces'),lit(1))\
.when(Business_REP.categories.contains('University Housing'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Religious Organizations
Business_REP = Business_REP\
.withColumn("Religious Organizations", when(Business_REP.categories.contains('Buddhist Temples'),lit(1))\
.when(Business_REP.categories.contains('Churches'),lit(1))\
.when(Business_REP.categories.contains('Hindu Temples'),lit(1))\
.when(Business_REP.categories.contains('Mosques'),lit(1))\
.when(Business_REP.categories.contains('Synagogues'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Restaurants
Business_REP = Business_REP\
.withColumn("Restaurants", when(Business_REP.categories.contains('Afghan'),lit(1))\
.when(Business_REP.categories.contains('African'),lit(1))\
.when(Business_REP.categories.contains('Senegalese'),lit(1))\
.when(Business_REP.categories.contains('South African'),lit(1))\
.when(Business_REP.categories.contains('American (New)'),lit(1))\
.when(Business_REP.categories.contains('American (Traditional)'),lit(1))\
.when(Business_REP.categories.contains('Arabian'),lit(1))\
.when(Business_REP.categories.contains('Argentine'),lit(1))\
.when(Business_REP.categories.contains('Armenian'),lit(1))\
.when(Business_REP.categories.contains('Asian Fusion'),lit(1))\
.when(Business_REP.categories.contains('Australian'),lit(1))\
.when(Business_REP.categories.contains('Austrian'),lit(1))\
.when(Business_REP.categories.contains('Bangladeshi'),lit(1))\
.when(Business_REP.categories.contains('Barbeque'),lit(1))\
.when(Business_REP.categories.contains('Basque'),lit(1))\
.when(Business_REP.categories.contains('Belgian'),lit(1))\
.when(Business_REP.categories.contains('Brasseries'),lit(1))\
.when(Business_REP.categories.contains('Brazilian'),lit(1))\
.when(Business_REP.categories.contains('Breakfast & Brunch'),lit(1))\
.when(Business_REP.categories.contains('British'),lit(1))\
.when(Business_REP.categories.contains('Buffets'),lit(1))\
.when(Business_REP.categories.contains('Burgers'),lit(1))\
.when(Business_REP.categories.contains('Burmese'),lit(1))\
.when(Business_REP.categories.contains('Cafes'),lit(1))\
.when(Business_REP.categories.contains('Cafeteria'),lit(1))\
.when(Business_REP.categories.contains('Cajun/Creole'),lit(1))\
.when(Business_REP.categories.contains('Cambodian'),lit(1))\
.when(Business_REP.categories.contains('Caribbean'),lit(1))\
.when(Business_REP.categories.contains('Dominican'),lit(1))\
.when(Business_REP.categories.contains('Haitian'),lit(1))\
.when(Business_REP.categories.contains('Puerto Rican'),lit(1))\
.when(Business_REP.categories.contains('Trinidadian'),lit(1))\
.when(Business_REP.categories.contains('Catalan'),lit(1))\
.when(Business_REP.categories.contains('Cheesesteaks'),lit(1))\
.when(Business_REP.categories.contains('Chicken Wings'),lit(1))\
.when(Business_REP.categories.contains('Chinese'),lit(1))\
.when(Business_REP.categories.contains('Cantonese'),lit(1))\
.when(Business_REP.categories.contains('Dim Sum'),lit(1))\
.when(Business_REP.categories.contains('Shanghainese'),lit(1))\
.when(Business_REP.categories.contains('Szechuan'),lit(1))\
.when(Business_REP.categories.contains('Comfort Food'),lit(1))\
.when(Business_REP.categories.contains('Creperies'),lit(1))\
.when(Business_REP.categories.contains('Cuban'),lit(1))\
.when(Business_REP.categories.contains('Czech'),lit(1))\
.when(Business_REP.categories.contains('Delis'),lit(1))\
.when(Business_REP.categories.contains('Diners'),lit(1))\
.when(Business_REP.categories.contains('Ethiopian'),lit(1))\
.when(Business_REP.categories.contains('Fast Food'),lit(1))\
.when(Business_REP.categories.contains('Filipino'),lit(1))\
.when(Business_REP.categories.contains('Fish & Chips'),lit(1))\
.when(Business_REP.categories.contains('Fondue'),lit(1))\
.when(Business_REP.categories.contains('Food Court'),lit(1))\
.when(Business_REP.categories.contains('Food Stands'),lit(1))\
.when(Business_REP.categories.contains('French'),lit(1))\
.when(Business_REP.categories.contains('Gastropubs'),lit(1))\
.when(Business_REP.categories.contains('German'),lit(1))\
.when(Business_REP.categories.contains('Gluten-Free'),lit(1))\
.when(Business_REP.categories.contains('Greek'),lit(1))\
.when(Business_REP.categories.contains('Halal'),lit(1))\
.when(Business_REP.categories.contains('Hawaiian'),lit(1))\
.when(Business_REP.categories.contains('Himalayan/Nepalese'),lit(1))\
.when(Business_REP.categories.contains('Hot Dogs'),lit(1))\
.when(Business_REP.categories.contains('Hot Pot'),lit(1))\
.when(Business_REP.categories.contains('Hungarian'),lit(1))\
.when(Business_REP.categories.contains('Iberian'),lit(1))\
.when(Business_REP.categories.contains('Indian'),lit(1))\
.when(Business_REP.categories.contains('Indonesian'),lit(1))\
.when(Business_REP.categories.contains('Irish'),lit(1))\
.when(Business_REP.categories.contains('Italian'),lit(1))\
.when(Business_REP.categories.contains('Japanese'),lit(1))\
.when(Business_REP.categories.contains('Korean'),lit(1))\
.when(Business_REP.categories.contains('Kosher'),lit(1))\
.when(Business_REP.categories.contains('Laotian'),lit(1))\
.when(Business_REP.categories.contains('Latin American'),lit(1))\
.when(Business_REP.categories.contains('Columbian'),lit(1))\
.when(Business_REP.categories.contains('Salvadoran'),lit(1))\
.when(Business_REP.categories.contains('Venezuelan'),lit(1))\
.when(Business_REP.categories.contains('Live/Raw Food'),lit(1))\
.when(Business_REP.categories.contains('Malaysian'),lit(1))\
.when(Business_REP.categories.contains('Meditteranean'),lit(1))\
.when(Business_REP.categories.contains('Mexican'),lit(1))\
.when(Business_REP.categories.contains('Middle Eastern'),lit(1))\
.when(Business_REP.categories.contains('Egyptian'),lit(1))\
.when(Business_REP.categories.contains('Lebanese'),lit(1))\
.when(Business_REP.categories.contains('Modern European'),lit(1))\
.when(Business_REP.categories.contains('Mongolian'),lit(1))\
.when(Business_REP.categories.contains('Pakistani'),lit(1))\
.when(Business_REP.categories.contains('Persian/Iranian'),lit(1))\
.when(Business_REP.categories.contains('Peruvian'),lit(1))\
.when(Business_REP.categories.contains('Pizza'),lit(1))\
.when(Business_REP.categories.contains('Polish'),lit(1))\
.when(Business_REP.categories.contains('Portuguese'),lit(1))\
.when(Business_REP.categories.contains('Russian'),lit(1))\
.when(Business_REP.categories.contains('Salad'),lit(1))\
.when(Business_REP.categories.contains('Sandwiches'),lit(1))\
.when(Business_REP.categories.contains('Scandinavian'),lit(1))\
.when(Business_REP.categories.contains('Scottish'),lit(1))\
.when(Business_REP.categories.contains('Seafood'),lit(1))\
.when(Business_REP.categories.contains('Singaporean'),lit(1))\
.when(Business_REP.categories.contains('Slovakian'),lit(1))\
.when(Business_REP.categories.contains('Soul Food'),lit(1))\
.when(Business_REP.categories.contains('Soup'),lit(1))\
.when(Business_REP.categories.contains('Southern'),lit(1))\
.when(Business_REP.categories.contains('Spanish'),lit(1))\
.when(Business_REP.categories.contains('Steakhouses'),lit(1))\
.when(Business_REP.categories.contains('Sushi Bars'),lit(1))\
.when(Business_REP.categories.contains('Taiwanese'),lit(1))\
.when(Business_REP.categories.contains('Tapas Bars'),lit(1))\
.when(Business_REP.categories.contains('Tapas/Small Plates'),lit(1))\
.when(Business_REP.categories.contains('Tex-Mex'),lit(1))\
.when(Business_REP.categories.contains('Thai'),lit(1))\
.when(Business_REP.categories.contains('Turkish'),lit(1))\
.when(Business_REP.categories.contains('Ukranian'),lit(1))\
.when(Business_REP.categories.contains('Vegan'),lit(1))\
.when(Business_REP.categories.contains('Vegetarian'),lit(1))\
.when(Business_REP.categories.contains('Vietnamese'),lit(1))\
.otherwise(0))

# COMMAND ----------

#Encoding Shopping
Business_REP = Business_REP\
.withColumn("Shopping", when(Business_REP.categories.contains('Adult'),lit(1))\
.when(Business_REP.categories.contains('Antiques'),lit(1))\
.when(Business_REP.categories.contains('Art Galleries'),lit(1))\
.when(Business_REP.categories.contains('Arts & Crafts'),lit(1))\
.when(Business_REP.categories.contains('Art Supplies'),lit(1))\
.when(Business_REP.categories.contains('Cards & Stationery'),lit(1))\
.when(Business_REP.categories.contains('Costumes'),lit(1))\
.when(Business_REP.categories.contains('Fabric Stores'),lit(1))\
.when(Business_REP.categories.contains('Framing'),lit(1))\
.when(Business_REP.categories.contains('Auction Houses'),lit(1))\
.when(Business_REP.categories.contains('Baby Gear & Furniture'),lit(1))\
.when(Business_REP.categories.contains('Bespoke Clothing'),lit(1))\
.when(Business_REP.categories.contains('Books, Mags, Music & Video'),lit(1))\
.when(Business_REP.categories.contains('Bookstores'),lit(1))\
.when(Business_REP.categories.contains('Comic Books'),lit(1))\
.when(Business_REP.categories.contains('Music & DVDs'),lit(1))\
.when(Business_REP.categories.contains('Newspapers & Magazines'),lit(1))\
.when(Business_REP.categories.contains('Videos & Video Game Rental'),lit(1))\
.when(Business_REP.categories.contains('Vinyl Records'),lit(1))\
.when(Business_REP.categories.contains('Bridal'),lit(1))\
.when(Business_REP.categories.contains('Computers'),lit(1))\
.when(Business_REP.categories.contains('Cosmetics & Beauty Supply'),lit(1))\
.when(Business_REP.categories.contains('Department Stores'),lit(1))\
.when(Business_REP.categories.contains('Discount Store'),lit(1))\
.when(Business_REP.categories.contains('Drugstores'),lit(1))\
.when(Business_REP.categories.contains('Electronics Repair'),lit(1))\
.when(Business_REP.categories.contains('Eyewear & Opticians'),lit(1))\
.when(Business_REP.categories.contains('Fashion'),lit(1))\
.when(Business_REP.categories.contains('Accessories'),lit(1))\
.when(Business_REP.categories.contains("Children's Clothing"),lit(1))\
.when(Business_REP.categories.contains('Department Stores'),lit(1))\
.when(Business_REP.categories.contains('Formal Wear'),lit(1))\
.when(Business_REP.categories.contains('Hats'),lit(1))\
.when(Business_REP.categories.contains('Leather Goods'),lit(1))\
.when(Business_REP.categories.contains('Lingerie'),lit(1))\
.when(Business_REP.categories.contains('Maternity Wear'),lit(1))\
.when(Business_REP.categories.contains("Men's Clothing"),lit(1))\
.when(Business_REP.categories.contains('Plus Size Fashion'),lit(1))\
.when(Business_REP.categories.contains('Shoe Stores'),lit(1))\
.when(Business_REP.categories.contains('Sports Wear'),lit(1))\
.when(Business_REP.categories.contains('Surf Shop'),lit(1))\
.when(Business_REP.categories.contains('Swimwear'),lit(1))\
.when(Business_REP.categories.contains('Used, Vintage & Consignment'),lit(1))\
.when(Business_REP.categories.contains("Women's Clothing"),lit(1))\
.when(Business_REP.categories.contains('Fireworks'),lit(1))\
.when(Business_REP.categories.contains('Flea Markets'),lit(1))\
.when(Business_REP.categories.contains('Flowers & Gifts'),lit(1))\
.when(Business_REP.categories.contains('Cards & Stationery'),lit(1))\
.when(Business_REP.categories.contains('Florists'),lit(1))\
.when(Business_REP.categories.contains('Gift Shops'),lit(1))\
.when(Business_REP.categories.contains('Golf Equipment Shops'),lit(1))\
.when(Business_REP.categories.contains('Guns & Ammo'),lit(1))\
.when(Business_REP.categories.contains('Hobby Shops'),lit(1))\
.when(Business_REP.categories.contains('Home & Garden'),lit(1))\
.when(Business_REP.categories.contains('Appliances'),lit(1))\
.when(Business_REP.categories.contains('Furniture Stores'),lit(1))\
.when(Business_REP.categories.contains('Hardware Stores'),lit(1))\
.when(Business_REP.categories.contains('Home Décor'),lit(1))\
.when(Business_REP.categories.contains('Hot Tub & Pool'),lit(1))\
.when(Business_REP.categories.contains('Kitchen & Bath'),lit(1))\
.when(Business_REP.categories.contains('Mattresses'),lit(1))\
.when(Business_REP.categories.contains('Nurseries & Gardening'),lit(1))\
.when(Business_REP.categories.contains('Jewelry'),lit(1))\
.when(Business_REP.categories.contains('Knitting Supplies'),lit(1))\
.when(Business_REP.categories.contains('Luggage'),lit(1))\
.when(Business_REP.categories.contains('Medical Supplies'),lit(1))\
.when(Business_REP.categories.contains('Mobile Phones'),lit(1))\
.when(Business_REP.categories.contains('Motorcycle Gear'),lit(1))\
.when(Business_REP.categories.contains('Musical Instruments & Teachers'),lit(1))\
.when(Business_REP.categories.contains('Office Equipment'),lit(1))\
.when(Business_REP.categories.contains('Outlet Stores'),lit(1))\
.when(Business_REP.categories.contains('Pawn Shops'),lit(1))\
.when(Business_REP.categories.contains('Personal Shopping'),lit(1))\
.when(Business_REP.categories.contains('Photography Stores & Services'),lit(1))\
.when(Business_REP.categories.contains('Pool & Billiards'),lit(1))\
.when(Business_REP.categories.contains('Pop-up Shops'),lit(1))\
.when(Business_REP.categories.contains('Shopping Centers'),lit(1))\
.when(Business_REP.categories.contains('Sporting Goods'),lit(1))\
.when(Business_REP.categories.contains('Bikes'),lit(1))\
.when(Business_REP.categories.contains('Golf Equipment'),lit(1))\
.when(Business_REP.categories.contains('Outdoor Gear'),lit(1))\
.when(Business_REP.categories.contains('Sports Wear'),lit(1))\
.when(Business_REP.categories.contains('Thrift Stores'),lit(1))\
.when(Business_REP.categories.contains('Tobacco Shops'),lit(1))\
.when(Business_REP.categories.contains('Toy Stores'),lit(1))\
.when(Business_REP.categories.contains('Trophy Shops'),lit(1))\
.when(Business_REP.categories.contains('Uniforms'),lit(1))\
.when(Business_REP.categories.contains('Watches'),lit(1))\
.when(Business_REP.categories.contains('Wholesale Stores'),lit(1))\
.when(Business_REP.categories.contains('Wigs'),lit(1))\
.otherwise(0))

# COMMAND ----------

Business_REP.display()

# COMMAND ----------

stringIndexer_state = StringIndexer(inputCol="state", outputCol="state_cat")
stringIndexer_state.setHandleInvalid("error")
model_state = stringIndexer_state.fit(Business_REP)
model_state.setHandleInvalid("error")
td = model_state.transform(Business_REP)
stringIndexer_city = StringIndexer(inputCol="city", outputCol="city_cat")
stringIndexer_city.setHandleInvalid("error")
model_city = stringIndexer_city.fit(td)
model_city.setHandleInvalid("error")
td = model_city.transform(td)
td=td.na.fill(0)
td.display()

# COMMAND ----------

#Converting T/F to 1/0 and changing the dtype
td = td.withColumn('garage', when(col('garage')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('street', when(col('street')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('validated', when(col('validated')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('lot', when(col('lot')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('valet', when(col('valet')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('touristy', when(col('touristy')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('hipster', when(col('hipster')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('romantic', when(col('romantic')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('intimate', when(col('intimate')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('trendy', when(col('trendy')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('upscale', when(col('upscale')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('classy', when(col('classy')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('casual', when(col('casual')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('dessert', when(col('dessert')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('latenight', when(col('latenight')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('lunch', when(col('lunch')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('dinner', when(col('dinner')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('brunch', when(col('brunch')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('breakfast', when(col('breakfast')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('monday', when(col('monday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('tuesday', when(col('tuesday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('wednesday', when(col('wednesday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('thursday', when(col('thursday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('friday', when(col('friday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('saturday', when(col('saturday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('sunday', when(col('sunday')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('dj', when(col('dj')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('background_music', when(col('background_music')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('no_music', when(col('no_music')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('jukebox', when(col('jukebox')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('live', when(col('live')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('video', when(col('video')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('karaoke', when(col('karaoke')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('straightperms', when(col('straightperms')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('coloring', when(col('coloring')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('extensions', when(col('extensions')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('africanamerican', when(col('africanamerican')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('curly', when(col('curly')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('kids', when(col('kids')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('perms', when(col('perms')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('asian', when(col('asian')==True,lit(1)).otherwise(0).cast(IntegerType()))

td = td.withColumn('BikeParking', when(col('BikeParking')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('GoodForKids', when(col('GoodForKids')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('ByAppointmentOnly', when(col('ByAppointmentOnly')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('RestaurantsReservations', when(col('RestaurantsReservations')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('RestaurantsGoodForGroups', when(col('RestaurantsGoodForGroups')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('HasTV', when(col('HasTV')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('RestaurantsTakeOut', when(col('RestaurantsTakeOut')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('RestaurantsDelivery', when(col('RestaurantsDelivery')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('OutdoorSeating', when(col('OutdoorSeating')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('Caters', when(col('Caters')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('RestaurantsTableService', when(col('RestaurantsTableService')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('BusinessAcceptsCreditCards', when(col('BusinessAcceptsCreditCards')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('WheelchairAccessible', when(col('WheelchairAccessible')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('BusinessAcceptsBitcoin', when(col('BusinessAcceptsBitcoin')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('DogsAllowed', when(col('DogsAllowed')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('HappyHour', when(col('HappyHour')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('GoodForDancing', when(col('GoodForDancing')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('CoatCheck', when(col('CoatCheck')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('Smoking', when(col('Smoking')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('DriveThru', when(col('DriveThru')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('AcceptsInsurance', when(col('AcceptsInsurance')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('BYOBCorkage', when(col('BYOBCorkage')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('Corkage', when(col('Corkage')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('AgesAllowed', when(col('AgesAllowed')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('BYOB', when(col('BYOB')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('DietaryRestrictions', when(col('DietaryRestrictions')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('RestaurantsCounterService', when(col('RestaurantsCounterService')==True,lit(1)).otherwise(0).cast(IntegerType()))
td = td.withColumn('Open24Hours', when(col('Open24Hours')==True,lit(1)).otherwise(0).cast(IntegerType()))


td=td.na.fill(0)



# COMMAND ----------

td = td.withColumn('RestaurantsPriceRange2', when(col('RestaurantsPriceRange2').isNull(),lit(0)).otherwise(col('RestaurantsPriceRange2')).cast(IntegerType()))
td = td.withColumn('WiFi', when(col('WiFi').isNull(),lit(0)).otherwise(lit(1)).cast(IntegerType()))
td = td.withColumn('RestaurantsAttire', when(col('RestaurantsAttire')=="u'casual'",lit(1)).when(col('RestaurantsAttire')=="'casual'",lit(2)).when(col('RestaurantsAttire')=="u'dressy'",lit(3)).when(col('RestaurantsAttire')=="'dressy'",lit(4)).otherwise(lit(0)).cast(IntegerType()))
td = td.withColumn('RestaurantsTakeOut', when(col('RestaurantsTakeOut')==True,lit(1)).otherwise(lit(0)).cast(IntegerType()))
td = td.withColumn('NoiseLevel', when(col('NoiseLevel')=="u'quiet'",lit(1)).when(col('NoiseLevel')=="u'average'",lit(2)).when(col('NoiseLevel')=="u'loud'",lit(3)).otherwise(lit(0)).cast(IntegerType()))
td = td.withColumn('Alcohol', when(col('Alcohol')=="u'none'",lit(1)).when(col('Alcohol')=="'none'",lit(1)).when(col('Alcohol')=="u'beer_and_wine'",lit(2)).when(col('Alcohol')=="u'full_bar'",lit(3)).otherwise(lit(0)).cast(IntegerType()))


# COMMAND ----------

td = td.withColumn('H_M_S',when(col('H_Monday')==0,lit(0)).otherwise(split(td['H_Monday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Monday')==0,lit(0)).otherwise(split(td['H_Monday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Monday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Monday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))


td = td.withColumn('H_M_S',when(col('H_Tuesday')==0,lit(0)).otherwise(split(td['H_Tuesday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Tuesday')==0,lit(0)).otherwise(split(td['H_Tuesday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Tuesday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Tuesday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))


td = td.withColumn('H_M_S',when(col('H_Wednesday')==0,lit(0)).otherwise(split(td['H_Wednesday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Wednesday')==0,lit(0)).otherwise(split(td['H_Wednesday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Wednesday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Wednesday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))

td = td.withColumn('H_M_S',when(col('H_Thursday')==0,lit(0)).otherwise(split(td['H_Thursday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Thursday')==0,lit(0)).otherwise(split(td['H_Thursday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Thursday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Thursday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))

td = td.withColumn('H_M_S',when(col('H_Friday')==0,lit(0)).otherwise(split(td['H_Friday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Friday')==0,lit(0)).otherwise(split(td['H_Friday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Friday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Friday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))


td = td.withColumn('H_M_S',when(col('H_Saturday')==0,lit(0)).otherwise(split(td['H_Saturday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Saturday')==0,lit(0)).otherwise(split(td['H_Saturday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Saturday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Saturday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))

td = td.withColumn('H_M_S',when(col('H_Sunday')==0,lit(0)).otherwise(split(td['H_Sunday'], '-').getItem(0))) 
td = td.withColumn('H_M_E',when(col('H_Sunday')==0,lit(0)).otherwise(split(td['H_Sunday'], '-').getItem(1)))
td = td.withColumn('H_M_S_H',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(0)).cast(IntegerType())) 
td = td.withColumn('H_M_E_H',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(0)).cast(IntegerType()))
td = td.withColumn('H_M_S_M',when(col('H_M_S')==0,lit(0)).otherwise(split(td['H_M_S'], ':').getItem(1)).cast(IntegerType())) 
td = td.withColumn('H_M_E_M',when(col('H_M_E')==0,lit(0)).otherwise(split(td['H_M_E'], ':').getItem(1)).cast(IntegerType()))
td = td.withColumn('Sunday_DT',when(col('H_M_S_H')<12,lit(1)).when(col('H_M_S_H').between(12,18),lit(2)).otherwise(lit(3)).cast(IntegerType()))
td = td.withColumn('H_M_S_H',col('H_M_S_H')*60+col('H_M_S_M')) 
td = td.withColumn('H_M_E_H',col('H_M_E_H')*60+col('H_M_E_M'))
td = td.withColumn('Duration_Sunday',when((col('H_M_S_H')==0) &(col('H_M_E_H')==0),lit(24)).otherwise(abs(col('H_M_E_H')-col('H_M_S_H'))/60).cast(DoubleType()))
td=td.na.fill(0)

td = td\
.drop(col("H_M_S"))\
.drop(col("H_M_E"))\
.drop(col("H_M_S_H"))\
.drop(col("H_M_E_H"))\
.drop(col("H_M_S_M"))\
.drop(col("H_M_E_M"))\
.drop(col("H_Monday"))\
.drop(col("H_Tuesday"))\
.drop(col("H_Wednesday"))\
.drop(col("H_Thursday"))\
.drop(col("H_Friday"))\
.drop(col("H_Saturday"))\
.drop(col("H_Sunday"))


td.display()

# COMMAND ----------

td = td.withColumn('Monday_DT',when(col('Duration_Monday')==24,lit(4)).otherwise(col('Monday_DT')).cast(IntegerType()))
td = td.withColumn('Tuesday_DT',when(col('Duration_Tuesday')==24,lit(4)).otherwise(col('Tuesday_DT')).cast(IntegerType()))
td = td.withColumn('Wednesday_DT',when(col('Duration_Wednesday')==24,lit(4)).otherwise(col('Wednesday_DT')).cast(IntegerType()))
td = td.withColumn('Thursday_DT',when(col('Duration_Thursday')==24,lit(4)).otherwise(col('Thursday_DT')).cast(IntegerType()))
td = td.withColumn('Friday_DT',when(col('Duration_Friday')==24,lit(4)).otherwise(col('Friday_DT')).cast(IntegerType()))
td = td.withColumn('Saturday_DT',when(col('Duration_Saturday')==24,lit(4)).otherwise(col('Saturday_DT')).cast(IntegerType()))
td = td.withColumn('Sunday_DT',when(col('Duration_Sunday')==24,lit(4)).otherwise(col('Sunday_DT')).cast(IntegerType()))
td.display()

# COMMAND ----------

td_viz=td
td = td\
.drop(col("categories"))\
.drop(col("longitude"))\
.drop(col("latitude"))\
.drop(col("state"))\
.drop(col("city"))\
.drop(col("address"))\
.drop(col("postal_code"))\
.drop(col("name"))

td = td.withColumn('review_count',col('review_count').cast(IntegerType()))
td = td.withColumn('is_open',col('is_open').cast(IntegerType()))
td = td.withColumn('stars',col('stars').cast(DoubleType()))
td = td.withColumn('stars',col('stars')*2)
td = td.withColumn('stars',col('stars').cast(IntegerType()))
td.display()



# COMMAND ----------

# td = td\
# .drop(col("categories"))\
# .drop(col("longitude"))\
# .drop(col("latitude"))\
# .drop(col("state"))\
# .drop(col("city"))\
# .drop(col("address"))\
# .drop(col("postal_code"))\
# .drop(col("name"))

# td = td.withColumn('review_count',col('review_count').cast(IntegerType()))
# td = td.withColumn('is_open',col('is_open').cast(IntegerType()))
# td = td.withColumn('stars',col('stars').cast(DoubleType()))
# td = td.withColumn('stars',col('stars')*2)
# td = td.withColumn('stars',col('stars').cast(IntegerType()))
# td.display()

# COMMAND ----------

#Checking the shape of the df td
print((td.count(), len(td.columns)))

# COMMAND ----------

#nunique business id
print((td.count(), len(td.columns)))
td.select("business_id").distinct().count()

# COMMAND ----------

#td.select("*").write.save("/FileStore/tables/parsed_data/Business_end.json")

# COMMAND ----------

# MAGIC %md
# MAGIC # End OF Business table processing

# COMMAND ----------

#https://github.com/Night-Time-Lab/NightTimeLab/blob/master/TextClustering/tests/KMeans.ipynb

# COMMAND ----------

                                                          ##############  CHECKIN DF ##############
  
display(parsed_checkin_df)

# COMMAND ----------

from pyspark.sql.functions import date_format
from pyspark.sql.functions import hour

#Timestamp - new date 
parsed_checkin_df = parsed_checkin_df.withColumn("date_new",to_date("current_timestamp")) 

# Timestamp to time
parsed_checkin_df = parsed_checkin_df.withColumn("time", date_format('date', 'HH:mm:ss'))

#Hour of the checkin
parsed_checkin_df = parsed_checkin_df.withColumn('hour',hour(parsed_checkin_df.date))

# Getting time of the day based on the hour 
parsed_checkin_df = parsed_checkin_df.withColumn("time_of_the_day", when(parsed_checkin_df.hour< 12 ,'checkin_Morning')
                                                 .when(parsed_checkin_df.hour.between(12,18),'checkin_Afternoon').otherwise('checkin_Night'))

display(parsed_checkin_df)



# COMMAND ----------

#gather the distinct values of the time of the day
distinct_values = parsed_checkin_df.select("time_of_the_day")\
                    .distinct()\
                    .rdd\
                    .flatMap(lambda x: x).collect()

# COMMAND ----------

#for each of the gathered values create a new column (encoding)
for distinct_value in distinct_values:
    function = udf(lambda item: 
                   1 if item == distinct_value else 0, 
                   IntegerType())
    new_column_name = "time_of_the_day"+'_'+distinct_value
    parsed_checkin_df = parsed_checkin_df.withColumn(new_column_name, function(col("time_of_the_day")))
display(parsed_checkin_df)

# COMMAND ----------

#Aggregating the created variables - one record per business
parsed_checkin_df = parsed_checkin_df.groupBy("business_id") \
                                                           .agg(min("date").alias("min_checkin"), \
                                                            max("date").alias("max_checkin"), \
                                                            count("date").alias("number_of_checkins"), \
                                                            sum("time_of_the_day_checkin_Night").alias("count_checkin_Night"), \
                                                            sum("time_of_the_day_checkin_Afternoon").alias("count_checkin_Afternoon"), \
                                                            sum("time_of_the_day_checkin_Morning").alias("count_checkin_Morning"), \
                                                            
         
     )

# COMMAND ----------

display(parsed_checkin_df)

# COMMAND ----------

parsed_checkin_df = parsed_checkin_df.select("*",
      datediff(col("min_checkin"),col("max_checkin")).alias("op_duration")
    )

# COMMAND ----------

display(parsed_checkin_df)

# COMMAND ----------

print((parsed_checkin_df.count(), len(parsed_checkin_df.columns)))

# COMMAND ----------

                                                     ######## REVIEW DF ########
display(parsed_review_df)

# COMMAND ----------

#Renaming the review features 
parsed_review_df = parsed_review_df.withColumnRenamed("stars","review_stars")
parsed_review_df = parsed_review_df.withColumnRenamed("cool","review_cool")
parsed_review_df = parsed_review_df.withColumnRenamed("useful","review_useful")
parsed_review_df = parsed_review_df.withColumnRenamed("funny","review_funny")

# COMMAND ----------

#Aggregating at business level
parsed_review_df = parsed_review_df.groupBy("business_id") \
                                                           .agg(min("date").alias("min_reviewdate"), \
                                                            max("date").alias("max_review_date"), \
                                                            count("review_id").alias("number_of_reviews"), \
                                                            sum("review_cool").alias("sum_cool"), \
                                                            sum("review_funny").alias("sum_funny"), \
                                                            sum("review_useful").alias("sum_useful"), \
                                                            mean("review_stars").alias("avg_stars"), \
                                                            
         
     )

# COMMAND ----------

display(parsed_review_df)

# COMMAND ----------

print((parsed_review_df.count(), len(parsed_review_df.columns)))

# COMMAND ----------

                                                               ############## TIP DF ##############
display(parsed_tip_df)

# COMMAND ----------

#Aggregating at Business level for created features
parsed_tip_df = parsed_tip_df.groupBy("business_id") \
    .agg(min("date").alias("min_date_tip"), \
         max("date").alias("max_date_tip"), \
         sum("compliment_count").alias("compliment_count_tip"), \
         count("user_id").alias("count_of_users_tip") \
         
     ) \
    

# COMMAND ----------

display(parsed_tip_df)

# COMMAND ----------

print((parsed_tip_df.count(), len(parsed_tip_df.columns)))

# COMMAND ----------

                                                     #####################COVID DF################

display(parsed_covid_df)

# COMMAND ----------

#Converting Target to 0, 1
parsed_covid_df = parsed_covid_df.withColumn('delivery or takeout', when(col('delivery or takeout')==True,lit(1)).otherwise(0).cast(IntegerType()))

# COMMAND ----------

#Considering Business id and delivery or takeout from Covid data
parsed_covid_df_final = parsed_covid_df.select("business_id","delivery or takeout")

# COMMAND ----------

display(parsed_covid_df_final)                                                        ######## REVIEW DF ########

# COMMAND ----------

print((parsed_covid_df_final.count(), len(parsed_covid_df_final.columns)))
parsed_covid_df_final.select("business_id").distinct().count()

# COMMAND ----------

parsed_covid_df_final=parsed_covid_df_final.dropDuplicates(["business_id"])

# COMMAND ----------

#parsed_covid_df_final = parsed_covid_df_final.distinct()

# COMMAND ----------

display(parsed_covid_df_final)

# COMMAND ----------

# MERGE BUSINESS AND COVID 

basetable  = parsed_covid_df_final.join(td,parsed_covid_df_final.business_id ==  td.business_id,"left").drop(td.business_id)
print((basetable.count(), len(basetable.columns)))

# COMMAND ----------

#MERGE CHECKINS WITH BASETABLE

basetable  = basetable.join(parsed_checkin_df,basetable.business_id ==  parsed_checkin_df.business_id,"left").drop(parsed_checkin_df.business_id)
print((basetable.count(), len(basetable.columns)))

# COMMAND ----------

#MERGE REVIEWS WITH BASETABLE

basetable  = basetable.join(parsed_review_df,basetable.business_id ==  parsed_review_df.business_id,"left").drop(parsed_review_df.business_id)
print((basetable.count(), len(basetable.columns)))

# COMMAND ----------

#MERGE TIP WITH BASETABLE

basetable  = basetable.join(parsed_tip_df,basetable.business_id ==  parsed_tip_df.business_id,"left").drop(parsed_tip_df.business_id)
print((basetable.count(), len(basetable.columns)))

# COMMAND ----------

display(basetable)

# COMMAND ----------

#Dropping unnecessary columns 
cols = ("min_checkin","max_checkin","min_reviewdate","max_review_date","min_date_tip","max_date_tip")

basetable = basetable.drop(*cols)

# COMMAND ----------

print((basetable.count(), len(basetable.columns)))

# COMMAND ----------

#CHECKING FOR NULL VALUES IN BASETABLE
from pyspark.sql.functions import col,isnan, when, count
a = basetable.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in basetable.columns]
   )

# COMMAND ----------

display(a)

# COMMAND ----------

#Filling NAs with 0 in Basetable
basetable = basetable.fillna(0)

# COMMAND ----------

#basetable.dtypes

# COMMAND ----------

#Changing the Datatypes of few columns to int
basetable = basetable.withColumn('number_of_checkins', col('number_of_checkins').cast(IntegerType()))
basetable = basetable.withColumn('count_checkin_Night', col('count_checkin_Night').cast(IntegerType()))
basetable = basetable.withColumn('count_checkin_Afternoon', col('count_checkin_Afternoon').cast(IntegerType()))
basetable = basetable.withColumn('count_checkin_Morning', col('count_checkin_Morning').cast(IntegerType()))
basetable = basetable.withColumn('number_of_reviews', col('number_of_reviews').cast(IntegerType()))
basetable = basetable.withColumn('sum_cool', col('sum_cool').cast(IntegerType()))
basetable = basetable.withColumn('sum_funny', col('sum_funny').cast(IntegerType()))
basetable = basetable.withColumn('sum_useful', col('sum_useful').cast(IntegerType()))
basetable = basetable.withColumn('compliment_count_tip', col('compliment_count_tip').cast(IntegerType()))
basetable = basetable.withColumn('count_of_users_tip', col('count_of_users_tip').cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ML MODELS

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
#from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

spark=SparkSession.builder \
.master ("local[*]")\
.appName("part3")\
.getOrCreate()

# COMMAND ----------

sc=spark.sparkContext
sqlContext=SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC https://github.com/gogundur/Pyspark-Logistic-Regression/blob/master/Pyspark/Pyspark%20Classification.ipynb

# COMMAND ----------

import pandas as pd
pd.set_option('display.max_colwidth', 80)
pd.set_option('max_columns', 12)

# COMMAND ----------

from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)

# COMMAND ----------

basetable.groupby('delivery or takeout').count().show()

# COMMAND ----------

#Renaming columns and removing spaces
basetable=basetable.withColumnRenamed("Active Life","Active_Life")
basetable=basetable.withColumnRenamed("Beauty & Spas","Beauty_Spas")
basetable=basetable.withColumnRenamed("Event Planning & Services","Event_Planning_Services")
basetable=basetable.withColumnRenamed("Financial Services","Financial_Services")
basetable=basetable.withColumnRenamed("Hotels & Travel","Hotels_Travel")
basetable=basetable.withColumnRenamed("Local Flavor","Local_Flavor")
basetable=basetable.withColumnRenamed("Local Services","Local_Services")
basetable=basetable.withColumnRenamed("Mass Media","Mass_Media")
basetable=basetable.withColumnRenamed("Professional Services","Professional_Services")
basetable=basetable.withColumnRenamed("Public Services & Government","Public_Services_Government")
basetable=basetable.withColumnRenamed("Real Estate","Real_Estate")
basetable=basetable.withColumnRenamed("Religious Organizations","Religious_Organizations")
basetable=basetable.withColumnRenamed("delivery or takeout","Target")
basetable=basetable.withColumnRenamed("Arts & Entertainment","Arts_Entertainment")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
#from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler,StandardScaler

# COMMAND ----------

import pandas as pd

# COMMAND ----------

assembler = VectorAssembler()\
         .setInputCols (['stars','review_count','is_open','BikeParking','GoodForKids','ByAppointmentOnly','RestaurantsPriceRange2','WiFi','RestaurantsAttire',\
 'NoiseLevel','RestaurantsReservations','RestaurantsGoodForGroups','HasTV','Alcohol','OutdoorSeating','Caters',\
 'RestaurantsTableService','BusinessAcceptsCreditCards','WheelchairAccessible','BusinessAcceptsBitcoin','DogsAllowed','HappyHour','GoodForDancing','CoatCheck',\
 'Smoking','DriveThru','AcceptsInsurance','BYOBCorkage','Corkage','AgesAllowed','BYOB','DietaryRestrictions','RestaurantsCounterService','Open24Hours','garage',\
 'street','validated','lot','valet','touristy','hipster', 'romantic','intimate','trendy','upscale','classy','casual','dessert','latenight','lunch','dinner',\
 'brunch','breakfast','monday','tuesday','wednesday','thursday','friday','saturday','sunday','dj','background_music','no_music','jukebox','live','video',\
 'karaoke','straightperms','coloring','extensions','africanamerican','curly','kids','perms','asian','Active_Life','Arts_Entertainment','Automotive','Beauty_Spas',\
 'Event_Planning_Services','Financial_Services','Hotels_Travel','Local_Flavor','Local_Services','Mass_Media','Nightlife','Pets','Professional_Services',\
                         'Public_Services_Government','Real_Estate','Religious_Organizations','Restaurants','Shopping','state_cat','city_cat','Monday_DT',\
                         'Duration_Monday','Tuesday_DT','Duration_Tuesday','Wednesday_DT','Duration_Wednesday','Thursday_DT','Duration_Thursday',\
                         'Friday_DT','Duration_Friday','Saturday_DT','Duration_Saturday','Sunday_DT','Duration_Sunday','number_of_checkins',\
                         'count_checkin_Night','count_checkin_Afternoon','count_checkin_Morning','op_duration','number_of_reviews','sum_cool',\
                         'sum_funny','sum_useful','avg_stars','compliment_count_tip','count_of_users_tip'])\
         .setOutputCol ("features")
        

assembler_df=assembler.transform(basetable)
assembler_df.toPandas().head()

# COMMAND ----------

#Splitting the data into Train and Test(20%)
train, test = assembler_df.randomSplit([0.8, 0.2], seed = 613)
print("Train Dataset Count: " + str(train.count()))
print("Test  Dataset Count: " + str(test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC LOGISTIC REGRESSION

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'Target', maxIter=5)
lrModel           = lr.fit(train)
predictions_test  = lrModel.transform(test)
predictions_train = lrModel.transform(train)

predictions_test.select('Target', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------



# COMMAND ----------

#LOGISTIC REGRESSION ACCURACY TRAIN 
accuracy = predictions_train.filter(predictions_train.Target == predictions_train.prediction).count() / float(predictions_train.count())
print("Logistic Regression Train Accuracy is : ",accuracy)

#LOGISTIC REGRESSION ACCURACY TEST
accuracy = predictions_test.filter(predictions_test.Target == predictions_test.prediction).count() / float(predictions_test.count())
print("Logistic Regression Test  Accuracy is : ",accuracy)


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
#LOGISTIC REGRESSION TRAIN AUC
my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='Target')
predictions_train.select('Target','prediction')
AUC = my_eval.evaluate(predictions_train)
print("Logistic Regression Train AUC score is : ",AUC)

#LOGISTIC REGRESSION TEST AUC
my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='Target')
predictions_test.select('Target','prediction')
AUC = my_eval.evaluate(predictions_test)
print("Logistic Regression Test  AUC score is : ",AUC)

# COMMAND ----------

#Function for Confusion Matrix 
class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# COMMAND ----------

#TRAIN COFUSION MATRIX
y_true = predictions_train.select("Target")
y_true = y_true.toPandas()

y_pred = predictions_train.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Logistic Regression Confusion matrix')
plt.show()

# COMMAND ----------

#TEST CONFUSION MATRIX
y_true = predictions_test.select("Target")
y_true = y_true.toPandas()

y_pred = predictions_test.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Logistic Regression Confusion matrix')
plt.show()

# COMMAND ----------

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Logistic Regression areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC RANDOM FOREST

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
#RF TRAIN
stringIndexer = StringIndexer(inputCol="Target", outputCol="indexed")
si_model = stringIndexer.fit(train)
train_rf = si_model.transform(train)

#RF TEST
stringIndexer = StringIndexer(inputCol="Target", outputCol="indexed")
si_model = stringIndexer.fit(test)
test_rf = si_model.transform(test)

# COMMAND ----------

#Renaming Train and Test columns - Target
train_rf = train_rf.withColumnRenamed("Target","label")
test_rf  = test_rf.withColumnRenamed("Target","label")

# COMMAND ----------

#Classification
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(numTrees=10, maxDepth=5, labelCol="indexed", seed=42,
    leafCol="leafId")
#Fit RF Model
rfcModel = rf.fit(train_rf)

#Predictions
rf_predictions_train = rfcModel.transform(train_rf)
rf_predictions_test = rfcModel.transform(test_rf)

# COMMAND ----------

#TRAIN ACCURACY - RF
accuracy = rf_predictions_train.filter(rf_predictions_train.label == rf_predictions_train.prediction).count() / float(rf_predictions_train.count())
print("Random Forest Train Accuracy : ",accuracy)

#TEST ACCURACY - RF
accuracy = rf_predictions_test.filter(rf_predictions_test.label == rf_predictions_test.prediction).count() / float(rf_predictions_test.count())
print("Random Forest Test  Accuracy : ",accuracy)

# COMMAND ----------

#rf_predictions_train.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

#RF Train AUC
my_eval_rf = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
rf_predictions_train.select('label','prediction')
AUC_rf = my_eval_rf.evaluate(rf_predictions_train)
print("Random Forest Train AUC score is : ",AUC_rf)

#RF Test AUC
my_eval_rf = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
rf_predictions_test.select('label','prediction')
AUC_rf = my_eval_rf.evaluate(rf_predictions_test)
print("Random Forest Test  AUC score is : ",AUC_rf)

# COMMAND ----------

#RANDOM FOREST TEST Confusion Matrix
y_true = rf_predictions_test.select("label")
y_true = y_true.toPandas()

y_pred = rf_predictions_test.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Random Forest Confusion matrix')
plt.show()

# COMMAND ----------

#Random Forest Feature importance
rfcModel.featureImportances

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decission Tree

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol="Target", outputCol="indexed")
si_model = stringIndexer.fit(train)
train_dt = si_model.transform(train)


stringIndexer = StringIndexer(inputCol="Target", outputCol="indexed")
si_model = stringIndexer.fit(test)
test_dt = si_model.transform(test)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=10, labelCol="indexed", leafCol="leafId",maxBins=50)
dtModel              = dt.fit(train_dt)
dt_predictions_test  = dtModel.transform(test_dt)
dt_predictions_train = dtModel.transform(train_dt)
#dt_predictions_test.select('Target', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

#TRAIN ACCURACY - DT
accuracy = dt_predictions_train.filter(dt_predictions_train.Target == dt_predictions_train.prediction).count() / float(dt_predictions_train.count())
print("Decision Tree Train Accuracy : ",accuracy)

#TEST ACCURACY - DT
accuracy = dt_predictions_test.filter(dt_predictions_test.Target == dt_predictions_test.prediction).count() / float(dt_predictions_test.count())
print("Decision Tree Test  Accuracy : ",accuracy)

# COMMAND ----------

#TRAIN AUC
my_eval_dt = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Target')
dt_predictions_train.select('Target','prediction')
AUC_dt = my_eval_dt.evaluate(dt_predictions_train)
print("Decision Tree Train AUC score is : ",AUC_dt)

#TEST AUC 
my_eval_dt = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Target')
dt_predictions_test.select('Target','prediction')
AUC_dt = my_eval_dt.evaluate(dt_predictions_test)
print("Decision Tree Test  AUC score is : ",AUC_dt)

# COMMAND ----------

#TRAIN Predictions
y_true = dt_predictions_train.select("Target")
y_true = y_true.toPandas()

y_pred = dt_predictions_train.select("prediction")
y_pred = y_pred.toPandas()

#CONFUSION MATRIX  TRAIN
cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)

#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

# COMMAND ----------

#TEST Predictions
y_true = dt_predictions_test.select("Target")
y_true = y_true.toPandas()

y_pred = dt_predictions_test.select("prediction")
y_pred = y_pred.toPandas()

#CONFUSION MATRIX  TEST
cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)

#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Decision Tree Confusion matrix')
plt.show()

# COMMAND ----------

dtModel.featureImportances


# COMMAND ----------

assembler2 = VectorAssembler()\
         .setInputCols (['stars','review_count','is_open','BikeParking','GoodForKids','ByAppointmentOnly','RestaurantsPriceRange2','WiFi','RestaurantsAttire',\
 'NoiseLevel','RestaurantsReservations','RestaurantsGoodForGroups','HasTV','Alcohol','OutdoorSeating','Caters',\
 'RestaurantsTableService','BusinessAcceptsCreditCards','WheelchairAccessible','count_of_users_tip','Active_Life','Nightlife',
'Arts_Entertainment','Beauty_Spas','lunch','RestaurantsTakeOut','RestaurantsDelivery'])\
         .setOutputCol ("features")

assembler_df2=assembler2.transform(basetable)
assembler_df2.toPandas().head()

# COMMAND ----------

#Splitting the data into Train and Test(20%)
train, test = assembler_df2.randomSplit([0.8, 0.2], seed = 613)
print("Train Dataset Count: " + str(train.count()))
print("Test  Dataset Count: " + str(test.count()))

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
#RF TRAIN
stringIndexer = StringIndexer(inputCol="Target", outputCol="indexed")
si_model = stringIndexer.fit(train)
train_rf = si_model.transform(train)

#RF TEST
stringIndexer = StringIndexer(inputCol="Target", outputCol="indexed")
si_model = stringIndexer.fit(test)
test_rf = si_model.transform(test)

# COMMAND ----------

#Renaming Train and Test columns - Target
train_rf = train_rf.withColumnRenamed("Target","label")
test_rf  = test_rf.withColumnRenamed("Target","label")

# COMMAND ----------

# #Exercise: Train a RandomForest model and tune the number of trees for values [150, 300, 500]
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# #Define pipeline
# rfc = RandomForestClassifier()
# rfPipe = Pipeline().setStages([rfc])

# #Set param grid
# rfParams = ParamGridBuilder()\
#   .addGrid(rfc.numTrees, [10])\
#   .build()

# rfCv = CrossValidator()\
#   .setEstimator(rfPipe)\
#   .setEstimatorParamMaps(rfParams)\
#   .setEvaluator(BinaryClassificationEvaluator())\
#   .setNumFolds(5) # Here: 5-fold cross validation



# COMMAND ----------

# #Run cross-validation, and choose the best set of parameters.
# rfcModel = rfCv.fit(train_rf)

# COMMAND ----------

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Evaluate model
rfevaluator = BinaryClassificationEvaluator()

# Create ParamGrid for Cross Validation
rfparamGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [5])
             .addGrid(rf.maxBins, [10])
             .addGrid(rf.numTrees, [5])
             .build())

# Create 10-fold CrossValidator
rfcv = CrossValidator(estimator = rf,
                      estimatorParamMaps = rfparamGrid,
                      evaluator = rfevaluator,
                      numFolds = 10)

# Run cross validations.
rfcvModel = rfcv.fit(train_rf)
print(rfcvModel)

# COMMAND ----------

print(rfcvModel)

# COMMAND ----------

#we are using this is for visualization 
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
parsed_covid_df_final_vz = parsed_covid_df.select("business_id","delivery or takeout","Virtual Services Offered")
parsed_covid_df_final_vz=parsed_covid_df_final_vz.dropDuplicates(["business_id"])
print((parsed_covid_df_final_vz.count(), len(parsed_covid_df_final_vz.columns)))
parsed_covid_df_final_vz.select("business_id").distinct().count()
basetable_vz = parsed_covid_df_final_vz.join(td_viz,parsed_covid_df_final_vz.business_id == td_viz.business_id,"left").drop(td_viz.business_id)
basetable_vz = basetable_vz.join(parsed_checkin_df,basetable_vz.business_id == parsed_checkin_df.business_id,"left").drop(parsed_checkin_df.business_id)
basetable_vz = basetable_vz.join(parsed_review_df,basetable_vz.business_id == parsed_review_df.business_id,"left").drop(parsed_review_df.business_id)
basetable_vz = basetable_vz.join(parsed_tip_df,basetable_vz.business_id == parsed_tip_df.business_id,"left").drop(parsed_tip_df.business_id)
basetable_vz.display()
