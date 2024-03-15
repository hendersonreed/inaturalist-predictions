# predicting unknown natural systems in Python with `tensorflow`

<div id="published-slug">published xx-xx-xxxx</div>

Using `pandas`, `tensorflow`, `scikit-learn`, and `plotly`.

There's a tendency towards epistemic arrogance in the hard sciences. There's a pervasive sense of overwhelming knowledge, just-finished progress, and a lack of things yet to be explored. This arrogance, this belief that there's little to understand isn't quite right though. Much is still being discovered. Biology and ecology in particular, due to the difficulties in setting up experiments and difficulties in measurement, have many *many* systems that remain mystifying.

For me, as an [amateur forager](/pages/foraging.html), I'm particularly interested in the why/how/when certain mushrooms begin to fruit. Turns out however that this isn't easy information to get authoritative information on. Some species aren't readily cultivatable even in lab settings, and so mycologists' only source of data regarding fruiting conditions is field observations. Experienced foragers begin to get a spidey-sense of when a particular species is about to fruit in a particular space, taking into account the fruiting of other associated species, rain fall, etc etc.

Luckily, there's [iNaturalist](https://www.inaturalist.org/home). A massive, crowdsourced, species observation database, iNaturalist has *lots* of data. For just my region, I downloaded around 200,000 observations of fungal fruiting bodies. So, I think I'll throw some Python at it and try to turn it into something useful!

## The data:

I exported more columns than I need for this MVP, so let's start small. This is what I have:

```
- id
- observed_on_string
- observed_on
- time_zone
- user_id
- quality_grade
- license
- url
- description
- num_identification_agreements
- num_identification_disagreements
- latitude
- longitude
- coordinates_obscured
- species_guess
- scientific_name
- common_name
- taxon_id
```

To begin with, I think `observed_on`, `species_guess`, `latitude`, and `longitude` are all that's necessary. With the resulting script, I'd like to be able to pass it a fragment of a row, and get some potential completions, e.g. pass it a date and a species name to get some potential locations to look at, or pass it a location and get some species and the dates they might be expected to be observed there.

One variable this MVP doesn't handle is weather data. The ideal would be to attach the preceding 7-14 days of weather history to each observation. Getting that weather data at the scale that I need it didn't seem easy/free after around an hour of googling, so I'm postponing that part of the project. This *does* mean that the results will be relying on data that's implicit in the observations (e.g. an appearance of *Cantherellus formosus* on Nov 16th, in L.L. Stub Stewart State Park occurred because of preceding weather, and so that weather is implicit in the observation), and as a result our trained model will not be able to map entirely accurately to the current year, because there's no weather data in our "prompt" ("prompt" here differs from what is typically called a "prompt" in the current language-model-focused understanding, but I don't personally know a better word.)

Anyways, a snippet of python produces our new CSV that contains just the columns for this MVP.

```python
import pandas

# I omitted some stuff around reading arguments, raising errors, etc.

df = pd.read_csv(input_file)

# Drop rows with missing values in any of the specified columns
df_filtered = df.dropna(subset=['observed_on', 'species_guess', 'latitude', 'longitude'])

# Keep only the desired columns
df_filtered = df_filtered[['observed_on', 'species_guess', 'latitude', 'longitude']]
df_filtered.to_csv(output_file, index=False)
```

We're down to just 132,225 observations, now that we've stripped out data that is missing any of those 4 elements.

## sketching out the interface:

Now, we need to build our neural network. To begin with, let's sketch out the ergonomics of our tool:

- I'd like a semi-generic tool, designed to train models on arbitrary CSV files that I can pass as a command-line argument.
- It should produce a model file that can be shared, versioned, etc etc.
- I should be able to use that model file with a second utility that produces N predictions of a CSV row, based on a fragment that I provide.
    - to accommodate a few different interaction modes, we might actually need multiple different models. One to allow predicting based on latitude/longitude pairs (i.e. predict what mushrooms *might* show up at a particular location) and another to predict based on date and species name (i.e. where might a particular mushroom species be found on a particular date.) Those are the first applications that spring to mind for me, but there are other potentially useful ones (like working out what dates a particular species might crop up at a particular location that is known to be a good foraging spot.)

Let's see how that might look in practice:

`csv-train.py our-training-data.csv` produces our model files.

`csv-predict.py model_file.h5 30 "2023-10-16, Cantherellus formosus"` produces a CSV file, which contains 30 predictions of locations that *Cantherellus formosus* might be found on October 16th.

That seems ergonomic enough for our use-cases at the moment. We'll know a lot more once we have the tool and begin using it. One potential adjustment would be to create a single command with subcommands, like `git`. That will be a better solution once we have to start worrying about the distribution of this tool, but for now I don't want to worry about combining these discrete parts (this is not always the right decision, some tools *must* be designed as complete units from the start - that's not the case here I'm fairly certain.)

## `csv-train.py`

This is relatively straightforward. First we'll build the model that targets the latitude/longitude pairs.

We need to do a bit of data processing to turn our data into something that a model can be trained on:

- We read the CSV file with `pandas` and turn the `latitude` and `longitude` columns into a tuple and make that our target column.
- `observed_on` comes to us as a string, so we'll convert it to a datetime object with the builtin `pandas.to_datetime` function.
- the `species_guess` items are categorical data, so we'll use scikit-learn's OneHotEncoder.

And... the OOM-killer decided our process was taking up too much memory and killed the process. So, perhaps some optimization is needed, or I need beefier harware (I guess I am just using my 7-year-old Dell XPS 13, so maybe that's only fair.)
