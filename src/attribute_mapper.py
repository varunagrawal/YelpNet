"""
Code to create a label file which should map each photo_id to a vector for each attribute of the corresponding business.,
"""

import json

attributes = {
    "Accepts Credit Cards": [False, True],
    "Alcohol": ["beer_and_wine", "full_bar", "none"],
    "Ambience": ["casual", "classy", "divey", "hipster", "intimate", "romantic", "touristy", "trendy", "upscale"],
    "Attire": ["casual", "dressy", "formal"],
    "Caters": [False, True],
    "Coat Check": [False, True],
    "Delivery": [False, True],
    "Dietary Restrictions": ["dairy-free", "gluten-free", "halal", "kosher", "soy-free", "vegan", "vegetarian"],
    "Drive-Thru": [False, True],
    "Good For": ["breakfast", "brunch", "dessert", "dinner", "latenight", "lunch"],
    "Good For Dancing": [False, True],
    "Good For Groups": [False, True],
    "Good for Kids": [False, True],
    "Happy Hour": [False, True],
    "Has TV": [False, True],
    'Music': ['background_music', 'dj','jukebox', 'karaoke', 'live', 'video'],
    "Noise Level": ["average", "loud", "quiet", "very_loud"],
    "Outdoor Seating": [False, True],
    "Parking": ["garage", "lot", "street", "valet", "validated"],
    "Price Range": [1, 2, 3, 4],
    'Smoking': ['outdoor', 'yes', 'no'],
    "Take-out": [False, True],
    "Takes Reservations": [False, True],
    "Waiter Service": [False, True]
}

template = {
    "id": None,
    "Accepts Credit Cards": 0,
    "Alcohol_beer_and_wine": 0,
    "Alcohol_full_bar": 0,
    "Alcohol_none": 0,
    "Ambience_casual": 0,
    "Ambience_classy": 0,
    "Ambience_divey": 0,
    "Ambience_hipster": 0,
    "Ambience_intimate": 0,
    "Ambience_romantic": 0,
    "Ambience_touristy": 0,
    "Ambience_trendy": 0,
    "Ambience_upscale": 0,
    "Attire_casual": 0,
    "Attire_dressy": 0,
    "Attire_formal": 0,
    "Caters": 0,
    "Coat Check": 0,
    "Delivery": 0,
    "Dietary_Restrictions_dairy-free": 0,
    "Dietary_Restrictions_gluten-free": 0,
    "Dietary_Restrictions_halal": 0,
    "Dietary_Restrictions_kosher": 0,
    "Dietary_Restrictions_soy-free": 0,
    "Dietary_Restrictions_vegan": 0,
    "Dietary_Restrictions_vegetarian": 0,
    "Drive-Thru": 0,
    "Good_For_breakfast": 0,
    "Good_For_brunch": 0,
    "Good_For_dessert": 0,
    "Good_For_dinner": 0,
    "Good_For_latenight": 0,
    "Good_For_lunch": 0,
    "Good For Dancing": 0,
    "Good For Groups": 0,
    "Good for Kids": 0,
    "Happy Hour": 0,
    "Has TV": 0,
    "Music_background_music": 0,
    "Music_dj": 0,
    "Music_jukebox": 0,
    "Music_karaoke": 0,
    "Music_live": 0,
    "Music_video": 0,
    "Noise_Level_average": 0,
    "Noise_Level_loud": 0,
    "Noise_Level_quiet": 0,
    "Noise_Level_very_loud": 0,
    "Outdoor Seating": 0,
    "Parking_garage": 0,
    "Parking_lot": 0,
    "Parking_street": 0,
    "Parking_valet": 0,
    "Parking_validated": 0,
    "Price_Range_1": 0,
    "Price_Range_2": 0,
    "Price_Range_3": 0,
    "Price_Range_4": 0,
    "Smoking_outdoor": 0,
    "Smoking_yes": 0,
    "Smoking_no": 0,
    "Take-out": 0,
    "Takes Reservations": 0,
    "Waiter Service": 0,
}


business_json = "../data/dataset/yelp_academic_dataset_business.json"

# Open the JSON file
#businesses = json.load(open(business_json, 'r'))

# The JSON file is not actually saved as JSON (F**k you, Yelp)
# So we read the data and manually convert it to json
with open(business_json, 'r') as f:
    t = f.read()

x = t.split('\n')
# get rid of last line as it is blank
x = x[:-1]

businesses = []
for l in x:
    businesses.append(json.loads(l))

photo_2_business_json = "../data/photo_id_to_business_id.json"

new_business = []

for id, b in enumerate(businesses):
    print b["name"]
    # print sorted(b["attributes"])

    # A new Business attribute object initialized each time
    bus_attr = dict(template)

    # Set the ID of the business
    bus_attr["id"] = b["business_id"]

    for k in attributes:
        # Check if the attribute is present for the business
        if k in b["attributes"]:
            # Check for attributes whose values are JSON objects
            if k == "Ambience" or k == "Good For" or k == "Dietary Restrictions" or k == "Music" or k == "Parking":
                for kk in b["attributes"][k].keys():
                    if b["attributes"][k][kk] == True:
                        bus_attr["_".join(k.split()) + "_" + kk] = 1
            else:
                if isinstance(b["attributes"][k], bool):
                    # check if it is a boolean value, and set it to 0 or 1 (which is the same as the index in the attributes dict)
                    bus_attr[k] = attributes[k].index(b["attributes"][k])
                else:
                    # The value of the attribute is a direct value
                    # So we construct the key as per the template
                    key = k.split()
                    key.append(str(b["attributes"][k]))
                    key = "_".join(key)
                    bus_attr[key] = 1

    assert(len(bus_attr) == 65)
    
    new_business.append(bus_attr)


with open("../data/business_attributes.json", 'w') as op:
    json.dump(new_business, op, indent=4, sort_keys=True)
