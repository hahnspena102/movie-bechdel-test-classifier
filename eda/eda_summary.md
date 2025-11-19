# Final Project Data Gathering + EDA 

Hahns Pena hpena02@calpoly.edu

Ishwarya Namburu inamburu@calpoly.edu

### What is your dataset and why did you choose it?

Our project is based off the Bechdel Test. To pass the Bechdel Test, the movie has to have at least two named women in it who talk to each other about something besides a man. Our classifier would take a movie and use features of that movie to make the classification without actually using the script itself manually performing the test.

For our project, we've chosen a big movie dataset with over 45000 movies. 
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv 

For labeling reasons, we are using https://bechdeltest.com/ and their dataset so we have a bechdel rating.

We chose these datasets because they have a lot of movies and a lot of features. Our main concern was finding a dataset that labeled a large amount of movies. Thus, we needed to find a large dataset in the case we need to cut a good amount for labeling. With the bechdel and movie metadata datasets merged, we have 8000+ movies. It has very promising features such as "overview", "cast", and "tagline" which we plan on analyzing further within our EDA. This dataset also has a imdb_id column which is useful for merging with our ground truth dataset as well as our evluations.

As far as credibility goes, this Kaggle dataset is highly downloaded and highly rated. 

### What did you learn from your EDA?

**TF-IDF**
A major feature in our dataset that we wanted to look into was the words, specifically, the title, overview, and tagline. 

For movies that pass, common words are life, young, family, woman, and new. For movies that fail, common words are man, men, dead, night, story. Some of these word associations make sense, some more obvious than the other. However, a good amount of the words don't really have strong signals.

Unsuprisingly, love is at the top of both movies that pass and fail. It's possible that movies who have a love story sometimes don't have conversations between two women about not a man. 

Overall, these tfidf distributions could be useful for a classifier but there aren't very many strong signals going on within them. Nevertheless, there are some strong signals, specifically the words "man" and "girl" so we may be able to use this for our classifier.

![title_pass](figures/title_pass.png)
![tagline_pass](figures/tagline_pass.png)
![overview_pass](figures/overview_pass.png)


![title_fail](figures/title_fail.png)
![tagline_fail](figures/tagline_fail.png)
![overview_fail](figures/overview_fail.png)

**Ratings**
Another feature we decided to look into was ratings. We found that movies that pass the bechdel test tend to get marginally better ratings, while movies that fail the test don't. However, the distributions are not drastically different, so this may not be a good feature to rely on heavily.

![Ratings](figures/ratings_distribution.png)

**Genre**
First we took a look at the distribution of genres for all the movies. Drama and comedy seem are the most common genres for all movies. Passing movies outnumber failing movies in genres such as drama, comedy, romance, horror, family, fantasy, mystery, animation, music, and foreign. Failing movies outnumber passing movies in a smaller number of genres, including action, adventure, crime, science fiction, history, war, western, and documentary. This makes sense, because genres such as war, crime, and adventure tend to be more male-centric. 
![Genre distribution](figures/genres_distribution.png)


Since a movie has many genres, we decided to create a heatmap to see if there was a signal in the genre combinations. Red cells represent the passing movies, and can be seen the brightest for the combos drama and romance, drama and comedy, romance and drama, and romance and comedy. This makes sense because these tend to be genres with importance given to the female characters. Blue cells represent the failing movies, and can be seen the brightest for the combos action and thriller, action and drama, and action and crime. As mentioned before, these are male centric genres.
![genre heatmap](figures/genres_heatmap.png)

Overall, genre seems like it could be useful for a classifier as there are some strong signals.



### What issues or open questions remain?
This is a vast dataset, so there are some other features we want to take a look at, such as cast and crew. We believe there may be a good signal in these features, but are not sure what the best way to explore them is. We also want to take a look into the release year of the films, and see if our hypothesis of older movies being more likely to fail is true. Once we do a little more exploratory data analysis, we can then start experimenting with what features to choose.

