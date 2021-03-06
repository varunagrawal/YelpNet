CS 7476 Advanced Computer Vision
================================
Professor James Hays
Georgia Tech

Project : Scene Classification over Yelp Restaurant Data
Dataset: [Yelp Dataset Challenge](http://www.yelp.com/dataset_challenge)
[Yelp's Approach](http://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html)

https://github.com/BVLC/caffe/wiki/Borrowing-Weights-from-a-Pretrained-Network
http://stackoverflow.com/questions/32680860/caffe-with-multi-label-images


### Attributes

Here is a list of all the attributes for each business.

Sublists are values that attribute can take. Everything else takes boolean values.

- Accepts Credit Cards
- Alcohol
    * beer_and_wine
    * full_bar
    * none
- Ambience (dict of key-booleans)
    * casual
    * classy
    * divey
    * hipster
    * intimate
    * romantic
    * touristy
    * trendy
    * upscale
- Attire
    * casual
    * dressy
    * formal
- Caters
- Coat Check
- Delivery
- Dietary Restrictions (dict of key-booleans)
    * dairy-free
    * gluten-free
    * halal
    * kosher
    * soy-free
    * vegan
    * vegetarian
- Drive-Thru
- Good For (dict of key-booleans)
    * breakfast
    * brunch
    * dessert
    * dinner
    * latenight
    * lunch
- Good For Dancing
- Good For Groups
- Good for Kids
- Happy Hour
- Has TV
- Music (dict of key-booleans)
    * background_music
    * dj
    * jukebox
    * karaoke
    * live
    * video
- Noise Level
    * average
    * loud
    * quiet
    * very_loud
- Outdoor Seating
- Parking (dict of type-booleans)
    * garage
    * lot
    * street
    * valet
    * validated
- Price Range
    * 1
    * 2
    * 3
    * 4
- Smoking
    * yes
    * no
    * outdoor
- Take-out
- Takes Reservations
- Waiter Service


Extra Attributes
- Accepts Insurance (Not important)
- Ages Allowed
- BYOB
- BYOB/Corkage
- By Appointment
- Corkage
- Dogs Allowed
- Hair Types Specialized In
- Open 24 Hours
- Order at Counter
- Wheelchair Accessible
- Wifi
