#!/usr/bin/env python
# coding: utf-8

# # The Future of OA:  A large-scale analysis projecting Open Access publication and readership
# 
# 

# 
# **Heather Piwowar &#42;<sup>1</sup>, Jason Priem &#42;<sup>1</sup>, Richard Orr<sup>1</sup>**  
# 
# &#42; shared first authorship  
# <sup>1</sup>_Our Research (team@ourresearch.org)_
# 
# Preprint first submitted: October 6, 2019
# 

# ------
# **Summary**
# 
# *Will move the Summary ([Section 4.5](#section-4-5) right now) up to the top here.  Is at the bottom right now so it can produce the graphs in it using the code below :)*
# 
# 
# ------

# <a id="section-1"></a>
# ## 1. Introduction
# 

# The adoption of [open access (OA)](https://en.wikipedia.org/wiki/Open_access) publishing is changing scholarly communication. Predicting the future prevalence of OA is crucial for many stakeholders making decisions now, including:
# 
# -   libraries deciding which journals to subscribe to and how much they should pay
# 
# -   institutions and funders deciding what mandates they should adopt, and the implications of existing mandates
# 
# -   scholarly publishers deciding when to flip their business models to OA
# 
# -   scholarly societies deciding how best to serve their members.
# 
# Despite how useful OA prediction would be, only a few studies have made an attempt to empirically predict open access rates. Lewis (2012) extrapolated the rate at which [gold OA](https://en.wikipedia.org/wiki/Open_access#Gold_OA) would replace subscription-based publishing using a simple log linear extrapolation of gold vs subscription market share. Antelman (2017) used one empirically-derived growth rate for [green OA](https://en.wikipedia.org/wiki/Open_access#Green_OA) and another for all other kinds of OA combined. Both of these studies are based on data collected before 2012, and rely on relatively simple models. Moreover, these studies predict the number of papers that are OA. While this number is important, it is arguably less meaningful than the number of views that are OA, since this latter number describes the prevalence of OA as experienced by actual readers.

# This paper aims to address this gap in the literature. In it, we build a detailed model using data extrapolated from large and up-to-date Unpaywall dataset (https://unpaywall.org/).  We use the model to predict the number of articles that will be OA (including gold, green, hybrid, and bronze OA) over the next five years, and also use data from the Unpaywall browser add-on (https://unpaywall.org/products/extension) to predict the proportion of scholarly article views that will lead readers to OA articles over time.
# 
# This paper aims to provide models of OA growth, taking the following complexities into account:
# 
# -   some forms of OA include a delay between when a paper is first published and when it is first freely available
# 
# -   different forms of open access are being adopted at different rates
# 
# -   wide-sweeping policy changes, technical improvements, or cultural changes may cause disruptions in the growth rates of OA in the future

# <a id="section-2"></a>
# ## 2. Data

# In[1]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# hidden: code to import libraries, set up database connection, other initialization\nimport warnings\nwarnings.filterwarnings(\'ignore\')\n\nimport os\nimport sys\nimport datetime\nimport pandas as pd\nimport numpy as np\nimport scipy\nfrom scipy import signal\nfrom scipy.optimize import curve_fit\nfrom scipy.stats.distributions import t\nfrom matplotlib import pyplot as plt\nimport matplotlib as mpl\nfrom matplotlib import cm\nfrom matplotlib.colors import ListedColormap\nimport seaborn as sns\nfrom sqlalchemy import create_engine\nimport sqlalchemy\nimport psycopg2\nfrom datetime import timedelta\nfrom IPython.display import display, HTML, Markdown\nimport cache_magic\nfrom tabulate import tabulate\n\n# our database connection\nredshift_engine = create_engine(os.getenv("DATABASE_URL_REDSHIFT"))\n\n# graph style\nsns.set(style="ticks")\n\n# long print, wrap\npd.set_option(\'display.expand_frame_repr\', False)\n\n# read from file if available, else from db and save it in a file for next time\n# will also help have data files ready for archiving in zenodo \ndef read_from_file_or_db(varname, query, skip_cache=False):\n    filename = "data/{}.csv".format(varname)\n    my_dataframe = pd.DataFrame()\n    try:\n        if not skip_cache:\n            my_dataframe = pd.read_csv(filename)\n    except IOError:\n        pass\n    if my_dataframe.empty:\n        global redshift_engine\n        my_dataframe = pd.read_sql_query(sqlalchemy.text(query), redshift_engine)\n        my_dataframe.to_csv(filename, index=False)  # cache for the future\n\n    return my_dataframe.copy()\n\n\n# make figure captions work.  use like this: \n# make a code cell, and include\n#   register_new_figure("my-figure-anchor-name") \n# before you want to refer to a figure.  This is where the link will go to.\n# and then in text markdown to refer to the figure\n#   {{figure_link("my-figure-anchor-name")}}\n\nglobal figure_so_far\nglobal figure_numbers\nfigures_so_far = 1\nfigure_numbers = {}\n\n# inspired by https://github.com/l-althueser/nbindex-jupyter/blob/master/nbindex/numbered.py\ndef leave_figure_anchor(anchor_text):\n    key  = u"figure-{}".format(anchor_text)\n    """\n    Adds numbered named object HTML anchors. Link to them in MarkDown using: [to keyword 1](#keyword-1)\n    """\n    return display(HTML(\'\'\'<div id="%s"></div>\n    <script>\n    var key = "%s"\n    $("div").each(function(i){\n        if (this.id === key){\n            this.innerHTML = \'<a name="\' + key + \'"></a>\';\n        }\n    });\n    </script>\n    \'\'\' % (key,key)))\n\ndef register_new_figure(anchor_text):\n    global figures_so_far\n    global figure_numbers\n    if not anchor_text in figure_numbers:\n        figure_numbers[anchor_text] = figures_so_far\n        leave_figure_anchor(anchor_text)\n        figures_so_far += 1\n    return figure_numbers[anchor_text]\n\ndef figure_link(anchor_text=None):\n    if anchor_text:\n        template = "[Figure {figure_number}](#figure-{anchor_text})"\n        my_return = template.format(figure_number=figure_numbers[anchor_text], \n                                    anchor_text=anchor_text)\n    else:\n        my_return = figure_numbers\n    return my_return\n    \n\n# set up colors\noa_status_order = ["green", "gold", "hybrid", "bronze", "closed"]\noa_status_colors = ["green", "gold", "orange", "brown", "grey"]\noa_color_lookup = pd.DataFrame(data = {"name": oa_status_order, "color": oa_status_colors, "order": range(0, len(oa_status_order))})\nmy_cmap = sns.color_palette(oa_status_colors)\n\ngraph_type_order = ["green", "gold", "hybrid", "immediate_bronze", "delayed_bronze", "closed"]\ngraph_type_colors = ["green", "gold", "orange", "brown", "salmon", "gray"]\ngraph_type_lookup = pd.DataFrame(data = {"name": graph_type_order, "color": graph_type_colors, "order": range(0, len(graph_type_order))})\nmy_cmap_graph_type = sns.color_palette(graph_type_colors)\n\ngraph_type_colors_plus_biorxiv = ["lawngreen"] + graph_type_colors\ngraph_type_order_plus_biorxiv = ["biorxiv"] + graph_type_order\nplus_biorxiv_labels = [\n    "green (biorxiv)",\n    "green (other)",\n    "gold",\n    "hybrid",\n    "bronze (immediate)",\n    "bronze (delayed)",\n    "closed"\n]\ngraph_type_plus_biorxiv_lookup = pd.DataFrame(data = {"name": graph_type_order_plus_biorxiv, "color": graph_type_colors_plus_biorxiv, "order": range(0, len(graph_type_colors_plus_biorxiv))})\nmy_cmap_graph_type_plus_biorxiv = sns.color_palette(graph_type_colors_plus_biorxiv)')


# The data in this analysis comes from two sources: (1) the Unpaywall dataset and (2) the access logs of the Unpaywall web browser extension.
# 
# <a id="section-oa-vocab"></a>
# <a id="section-2-1"></a>
# ### 2.1 OA type: the Unpaywall dataset of OA availability 
# 
# Predicting levels of open access publication in the future requires detailed, accurate, timely data. This study uses the [Unpaywall](https://unpaywall.org/) dataset to provide this data. Unpaywall is an open source application that  links every research article that has been assigned a Crossref DOI (more than 100 million in total) to the OA URLs where the paper can be read for free. It  is built and maintained by Our Research (formerly Impactstory), a US-based nonprofit organization. Unpaywall gathers data gathered from over 50,000 journals and open-access repositories from all over the world. The full Unpaywall dataset is freely, publicly available (see details: <https://unpaywall.org/user-guides/research>).

# Our definitions of OA type (gold, green, hybrid, bronze, closed) are described in Piwowar et al. (2018). To facilitate prediction, for the purpose of this analysis we subdivided bronze OA into immediate and delayed OA. In summary, these definitions are:
# 
# -   **<span style="color:gold; font-size:100%;">&#x2588;</span>  Gold:** published in a fully-OA journal
# -   **<span style="color:orange; font-size:100%;">&#x2588;</span> Hybrid:** published in a toll-access journal, available on the publisher site, with an OA license
# -   **<span style="color:brown; font-size:100%;">&#x2588;</span> Bronze:** published in a toll-access journal, available on the publisher site, without an OA license
#     -   **<span style="color:brown; font-size:100%;">&#x2588;</span> Immediate Bronze:** available as Bronze OA immediately upon publication
#     -   **<span style="color:salmon; font-size:100%;">&#x2588;</span> Delayed Bronze:** available as Bronze OA after an embargo period
# -   **<span style="color:green; font-size:100%;">&#x2588;</span> Green:** published in a toll-access journal and the only fulltext copy available is in an OA repository
# -   **<span style="color:gray; font-size:100%;">&#x2588;</span> Closed:** everything else
# 
# This analysis uses all articles with a Crossref article type of "journal-article" published between 1950 and the date of the analysis (October 2019), which is 71 million articles. 

# <a id="section-2-2"></a>
# ### 2.2 Article views: access logs of the Unpaywall web browser extension
# 
# 
# Predicting the open access pattern of usage requests requires knowing the relative usage demands of papers based on their age. This study has extracted these pageview patterns from the usage logs of the [Unpaywall browser extension](https://unpaywall.org/products/extension) for Chrome and Firefox.

# In[2]:


register_new_figure("unpaywall_map");


# This extension is an open-source tool made by the same non-profit as the Unpaywall dataset described above, with the goal of helping people conveniently find free copies of research papers directly from their web browser. The extension has [more than 200,000 active users](http://blog.our-research.org/unpaywall-200k-users/), distributed globally, as shown in {{ print figure_link("unpaywall_map") }}.
# 
# <img src="https://github.com/Impactstory/future-oa/blob/master/img/unpaywall%20extension%20users%20by%20location.jpg?raw=true"></img>
# 
# **{{ print figure_link("unpaywall_map") }}: Map of Unpaywall users in February 2019.**
# 
# 
# The Unpaywall browser automatically extension detects when a user is on a scholarly article webpage -- we consider this an access request, or a view. The extension can be disabled, or can be configured to only run upon request, but very few users use these settings. 
# 
# The extension received more than 3 million article access requests in July 2019 which we use for most of our analysis. Because readership data is private and potentially sensitive, we are not releasing the Unpaywall usage logs along with the other datasets behind this paper other than as aggregate counts by OA type by year.
# 

# <a id="section-3"></a>
# ## 3. Approach
# 
# <a id="section-3-1"></a>
# ### 3.1 Overview 
# 
# 
# The goal of this analysis is to predict two aspects of OA growth:
# 1. Growth in OA articles and their proportion of the literature over time
# 2. Growth in OA article views and their proportion of all literature views over time

# We examine the growth in OA articles *by date of observation*, rather than by date of publication.  This requires us to calculate the OA lag between publication and availability for different types of OA, which is done in [Section 4.1](#section-4-1).
# 
# Once we have the pattern of OA availability by year, we forecast the OA availability for future years by assuming that it will have the same overall pattern as previous years -- the papers that will be made available next year will have the same age distribution as papers that were made available last year.  We allow the absolute number of papers to increase year-over-year: we estimate the future growth multiplier by extrapolating the height of past availability curves.  This analysis is presented in [Section 4.2](#section-4-2).

# Next, we turn to predicting the growth of OA article views -- what proportion of what is read is available OA, and how will this change in the future?  The Unpaywall browser extension logs give us a relative baseline of what is read right now.  By assuming that reading patterns remain relatively unchanged over time (specifically the probability that a reader wants to read a paper given its age and OA type), we use the publication estimates we made in previous sections to calculate the relative number of views by OA type in the past and the future.  This is described in [Section 4.3](#section-4-3).
# 
# Finally, we look at the impact of extending the model to include a disruptive change, in this case the growth of bioRxiv, in [Section 4.4](#section-4-4).
# 

# <a id="section-3-2"></a>
# ### 3.2 Glossary
# 
# In addition to the OA types defined in [Section 2.1](#section-oa-vocab), we define additional terms as we use them in this paper, in approximate order they are discussed:
# 
# - **Date of publication**: the date an article is published in a journal
# - **Embargo**: the delay that some toll-access journals require between date of publication and when an article can be made Green or Delayed Bronze OA
# - **Self-archiving**: when an author posts their article in an OA repository
# - **OA type**: the OA classification of an article, as defined in [Section 2](#section-oa-vocab).  The OA type of an article may change over time (from Closed to Delayed Bronze OA, or from Closed to Green OA) because of embargoes and other self-archiving delays
# - **Date first available OA**: the date an article first becomes an OA type other than "Closed"
# - **OA lag**: the length of time between an article's Date of Publication and its Date First Available OA
# <pre></pre>

# - **OA assessment**: the determination of the OA type of an article at a given point in time
# - **Date of observation**: the point in time for which we make an OA assessment for an article.  Explained in [Section 3.3](#section-3-3).
# - **Observation age** of an article: the length of time between an OA assessment observation and the article's date of publication
# <pre></pre>

# - **View**: someone on the internet visited the publisher webpage of an article, presumably with the hope of reading the article
# - **Date of view**: the date of the view
# - **View age** of an article: the length of time between an article's date of publication and the date of a view
# <pre></pre>

# - **Articles by age curve**: for a given snapshot, the plot of snapshot age (in years) on the x-axis and number of articles published of that snapshot age on the y-axis
# - **Views by age curve**: the plot of view age (in years) on the x-axis and number of views received by articles of that view age on the y-axis
# - **Views per article by age curve**: the plot of view or snapshot age (in years) on the x-axis and number of views per article (by views of that view age and articles of that snapshot age) on the y-axis
# - **Views per year curve**:  the plot of year on the x-axis and the number of views estimated to have been made that year on the y-axis

# <a id="section-3-3"></a>
# ### 3.3 Date of Observation

# In[3]:


register_new_figure("date_of_observation");


# In this paper we approach the growth of OA from the Date of Observation of OA assessment, rather than the date of publication.  We explain this with the use of {{ print figure_link("date_of_observation") }}.
# 

# <img src="https://github.com/Impactstory/future-oa/blob/master/img/date_of_observation_prediction.jpg?raw=true" style="float:right;"></img>
# 
# **{{ print figure_link("date_of_observation") }}: Date of observation.**
# 
# 
# Let’s imagine two observers, <span style="color:blue">Alice</span> (blue) and <span style="color:red">Bob</span> (red), shown by the two stick figures at the top of {{ print figure_link("date_of_observation") }}.
# 
# Alice lives at the end of Year 1--that’s her "Date Of Observation." Looking down, she can see all 8 articles (represented by solid colored dots) published in Year 1, along with their access status: Gold OA, Green OA, or Closed. The Year of Publication for all eight of these articles is Year 1.
# 
# Alice likes reading articles, so she decides to read all eight Year 1 articles, one by one.
# 
# She starts with Article A. This article started its life early in the year as Closed. Later that year, though--after an OA Lag of about six months--Article A became Green OA as its author deposited a manuscript (the green circle) in their institutional repository. Now, at Alice’s Date of Observation, it’s open! Excellent. Since Alice is inclined toward organization, she puts Article A article in a stack of Green articles she’s keeping below.
# 
# Now let’s look at Bob. Bob lives in Alice’s future, in Year 3 (ie, his “Date of Observation” is Year 3). Like Alice, he’s happy to discover that Article A is open. He puts it in his stack of Green OA articles, which he’s further organized by date of their publication (it goes in the Year 1 stack).
# 
# Next, Alice and Bob come to Article B, which is a tricky one. Alice is sad: she can’t read the article, and places it in her Closed stack. Unbeknownst to poor Alice, she is a victim of OA Lag, since Article B will become OA in Year 2. By contrast, Bob, from his comfortable perch in the future, is able to read the article. He places it in his Green Year 1 stack. He now has two articles in this stack, since he’s found two Green OA articles in Year 1.
# 
# Finally, Alice and Bob both find Article C is closed, and place it in the closed stack for Year 1. We can model this behavior for a hypothetical reader at each year of observation, giving us their view on the world--and that’s exactly the approach we take in this paper.
# 
# Now, let’s say that Bob has decided he’s going to figure out what OA will look like in Year 4. He starts with Gold. This is easy, since Gold article are open immediately upon publication, and publication date is easy to find from article metadata. So, he figures out how many articles were Gold for Alice (1), how many in Year 2 (3), and how many in his own Year 3 (6). Then he computes percentages, and graphs them out using the stacked area chart at the bottom of {{ print figure_link("date_of_observation") }}. From there, it’s easy to extrapolate forward a year.
# 
# For Green, he does the same thing--but he makes sure to account for OA Lag. Bob is trying to draw a picture of the world every year, as it appeared to the denizens of that world. He wants Alice’s world as it appeared to Alice, and the same for Year 2, and so on. So he includes OA Lag in his calculations for Green OA, in addition to publication year. Once he has a good picture from each Date Of Observation, and a good understanding of what the OA Lag looks like, he can once again extrapolate to find Year 4 numbers.
# 
# Bob is using the same approach we will use in this paper--although in practice, we will find it to be rather more complex, due to varying lengths of OA Lag, additional colors, of OA, and a lack of stick figures. 
# 

# <a id="section-3-4"></a>
# ### 3.4 Statistical analysis
# 
# The analysis was implemented as an executable python Jupyter notebook using the pandas, scipy, matplotlib, and sqlalchemy libraries. See the [Data and code availability section](#Data-and-code-availability) below for links to the analysis code and raw data.

# *---- delete the text between these lines in the final paper ----*

# #### Code: Functions

# See notebook.

# In[4]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\ndef get_data_extrapolated(graph_type, data_type=False, extrap="linear", now_delta_years=0, cumulative=True):\n    \n    calc_min_year = 1951\n    display_min_year = 2010\n    now_year = 2019 - now_delta_years\n    max_year = 2024\n\n    min_y = 0\n    max_y = None\n    color = graph_type\n    if "bronze" in graph_type:\n        color = "bronze"\n                        \n    if isinstance(data_type, pd.DataFrame):\n        df_this_color = data_type.loc[(data_type.graph_type==graph_type)]\n    elif data_type == "basic":\n        df_this_color = articles_by_color_by_year.loc[(articles_by_color_by_year.oa_status==color)]\n    else:\n        df_this_color = articles_by_graph_type_by_year.loc[(unpaywall_graph_type.oa_status==graph_type)]\n\n    totals = pd.DataFrame()\n    for i, prediction_year in enumerate(range(calc_min_year, now_year)):\n\n        if "published_year" in df_this_color.columns:\n            if cumulative:\n                df_this_plot = df_this_color.loc[(df_this_color["published_year"] <= prediction_year)]\n            else:\n                df_this_plot = df_this_color.loc[(df_this_color["published_year"] == prediction_year)]\n        else:\n            df_this_plot = df_this_color\n        y = [a for a in df_this_plot["num_articles"] if not np.isnan(a)]\n        prediction_y = sum(y)\n\n        totals = totals.append(pd.DataFrame(data={"prediction_year": [prediction_year], \n                                             "num_articles": [prediction_y]}))\n\n      \n    x = totals["prediction_year"]\n    y = totals["num_articles"]\n    xnew = np.arange(now_year-1, max_year+1, 1)\n    if extrap=="linear":\n        f = scipy.interpolate.interp1d(x, y, fill_value="extrapolate", kind="linear")\n        ynew = f(xnew)\n    else:\n        f = scipy.interpolate.interp1d(x, np.log10(y), fill_value="extrapolate", kind="linear")\n        ynew = 10 ** f(xnew)\n    \n    new_data = pd.DataFrame({"color":color, "graph_type": graph_type, "x":np.append(x[:-1], xnew), "y":np.append(y[:-1], ynew)})\n\n    return new_data\n\n\ndef graph_data_extrapolated(graph_type, data_type=False, extrap="linear", now_delta_years=0, ax=None, cumulative=True):\n    calc_min_year = 1951\n    display_min_year = 2000\n    now_year = 2019 - now_delta_years\n    max_year = 2024\n\n    min_y = 0\n    max_y = None\n    color = graph_type\n    if "bronze" in graph_type:\n        color = "bronze"\n    \n    new_data = get_data_extrapolated(graph_type, data_type, extrap, now_delta_years, cumulative)\n\n    year_range = range(display_min_year, now_year)\n    \n    if not isinstance(data_type, pd.DataFrame) and data_type == "simple":\n        my_color_lookup = oa_color_lookup.loc[oa_color_lookup["name"]==color]\n    else:\n        my_color_lookup = graph_type_lookup.loc[graph_type_lookup["name"]==graph_type]\n    \n    if not ax:\n        fig = plt.figure()\n        ax = plt.subplot(111)\n\n    if not max_y:\n        max_y = 5 * max(new_data["y"])\n\n    df_actual = new_data.loc[new_data["x"] < now_year]\n    x = [int(a) for a in df_actual["x"]]\n    y = [int(a) for a in df_actual["y"]]\n    df_future = new_data.loc[new_data["x"] >= now_year]\n    xnew = [int(a) for a in df_future["x"]]\n    ynew = [int(a) for a in df_future["y"]]\n\n    ax.plot(x, y, \'o\', color="black")\n    ax.fill_between(x, y, color=my_color_lookup["color"])\n\n    ax.plot(xnew, ynew, \'o\', color="black", alpha=0.3)\n    ax.fill_between(xnew, ynew, color=my_color_lookup["color"], alpha=0.3)\n    if cumulative:\n        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: \'{0:,.0f}\'.format(y/(1000*1000.0))))\n        ax.set_ylabel("articles (millions)")\n        ax.set_xlabel("year")\n    else:\n        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: \'{0:,.1f}\'.format(y/(1000*1000.0))))\n        ax.set_ylabel("articles (millions)")\n        ax.set_xlabel("year of publication")\n    ax.set_xlim(min(year_range), max_year)\n    ax.set_title(graph_type);\n\n    return new_data')


# In[5]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# graph!  :)\n\ndef graph_available_papers_at_year_of_availability(graph_type, now_delta_years=0, ax=None):\n    calc_min_year = 1951\n    display_min_year = 2010\n    now_year = 2018 - now_delta_years\n    max_year = 2024\n\n    color = graph_type\n    if "bronze" in graph_type:\n        color = "bronze"\n\n    if graph_type == "biorxiv":\n        my_color_lookup = {"color": "limegreen"}\n    else:\n        my_color_lookup = graph_type_lookup.loc[graph_type_lookup["name"]==graph_type]        \n        \n    all_papers_per_year = get_papers_by_availability_year_including_future(graph_type, calc_min_year, max_year)\n\n    most_recent_year = all_papers_per_year.loc[all_papers_per_year.article_years_from_availability == 0]\n    \n    x = [int(a) for a in most_recent_year.loc[most_recent_year.prediction_year <= now_year]["prediction_year"]]\n    xnew = [int(a) for a in most_recent_year.loc[most_recent_year.prediction_year > now_year]["prediction_year"]]\n    y = [int(a) for a in most_recent_year.loc[most_recent_year.prediction_year <= now_year]["num_articles"]]\n    ynew = [int(a) for a in most_recent_year.loc[most_recent_year.prediction_year > now_year]["num_articles"]]\n\n    year_range = range(display_min_year, now_year)\n    if not ax:\n        fig = plt.figure()\n        ax = plt.subplot(111)\n\n    max_y = 1.2 * max(ynew)\n\n    ax.plot(x, y, \'o\', color="black")\n    ax.fill_between(x, y, color=my_color_lookup["color"])\n\n    ax.plot(xnew, ynew, \'o\', color="black", alpha=0.3)\n    ax.fill_between(xnew, ynew, color=my_color_lookup["color"], alpha=0.3)\n    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: \'{0:,.2f}\'.format(y/(1000*1000.0))))\n    ax.set_ylabel("total papers (millions)")\n\n    ax.set_xlim(min(year_range), max_year)\n#         ax.set_ylim(0, max_y)\n    ax.set_xlabel(\'year of observation\')\n    title = plt.suptitle("OA status by observation year")\n    title.set_position([.5, 1.05])\n    all_papers_per_year.reset_index(inplace=True)\n    return all_papers_per_year')


# In[6]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\ndef graph_available_papers_in_observation_year_by_pubdate(graph_type, data, observation_year, ax=None):\n    display_min_year = 2010\n    max_year = 2025\n\n    x = [int(a) for a in data["publication_date"]]\n    y = [int(a) for a in data["num_articles"]]\n\n    my_color_lookup = graph_type_lookup.loc[graph_type_lookup["name"]==graph_type]\n    if not ax:\n        fig = plt.figure()\n        ax = plt.subplot(111)\n\n    alpha = 1\n#     if observation_year > 2018:\n#         alpha = 0.3\n    ax.bar(x, y, color=my_color_lookup["color"], alpha=alpha, width=1, edgecolor=my_color_lookup["color"])\n\n    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: \'{0:,.1f}\'.format(y/(1000*1000.0))))\n    ax.set_xlim(display_min_year, max_year+1)\n    max_y = 1.2 * data.num_articles.max()\n    try:\n        ax.set_ylim(0, max_y)\n    except:\n        pass\n    ax.set_xlabel("")\n    ax.set_ylabel("")\n    ax.spines[\'top\'].set_visible(False)\n    ax.spines[\'right\'].set_visible(False)\n    \n#     ax.set_title("{}: {}".format(graph_type, observation_year));  \n#     title = plt.suptitle("Availability in {}, by publication date".format(observation_year))\n#     title.set_position([.5, 1.05])\n    return \n')


# In[7]:


def get_papers_by_availability_year_including_future(graph_type, start_year, end_year):
    start_calc_year = 2009
    last_year_before_extrap = 2017
    offset = 0
    global final_extraps
    
    my_return = pd.DataFrame()

    for prediction_year in range(min(start_year, start_calc_year), last_year_before_extrap+1):        
#         print prediction_year
        papers_per_year = get_papers_by_availability_year(graph_type, prediction_year, just_this_year=False)
        papers_per_year["prediction_year"] = prediction_year
        my_return = my_return.append(papers_per_year)
        
    if end_year >= last_year_before_extrap:
        scale_df = final_extraps.copy()
        current_year_all = get_papers_by_availability_year(graph_type, last_year_before_extrap, just_this_year=False)
        now_year_new = get_papers_by_availability_year(graph_type, last_year_before_extrap, just_this_year=True)
        for i, prediction_year in enumerate(range(last_year_before_extrap+1, end_year+1)): 
            current_year_all["article_years_from_availability"] += 1 
#             print now_year_all.head()
#             print now_year_new.head()
            merged_df = current_year_all.merge(now_year_new, on="article_years_from_availability", suffixes=["_all", "_new"], how="outer")
            merged_df = merged_df.fillna(0)
#             print merged_df.head(10)
            scale = float(scale_df.loc[(scale_df.x==prediction_year)&(scale_df.graph_type==graph_type)].y) / int(scale_df.loc[(scale_df.x==last_year_before_extrap)&(scale_df.graph_type==graph_type)].y) 
            merged_df["num_articles"] = merged_df["num_articles_all"] + [int(scale * a) for a in merged_df["num_articles_new"]]
            merged_df["prediction_year"] = prediction_year
            current_year_all = pd.DataFrame(merged_df, columns=["num_articles", 
                                                      "article_years_from_availability", 
                                                      "prediction_year"])
            my_return = my_return.append(current_year_all)

    my_return.reset_index(inplace=True)
    return my_return


# In[8]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# graph!  :)\n\ndef graph_views(graph_type, data=None, now_delta_years=0, ax=None):\n    calc_min_year = 1951\n    display_min_year = 2010\n    now_year = 2018 - now_delta_years\n    max_year = 2025\n\n    color = graph_type\n\n    if isinstance(data, pd.DataFrame):\n        df_views_by_year = data\n    else:\n        df_views_by_year = get_predicted_views(graph_type, display_min_year, max_year)\n\n    year_range = range(display_min_year, now_year)\n    if graph_type == "biorxiv":\n        my_color_lookup = {"color": "limegreen"}\n    else:\n        my_color_lookup = graph_type_lookup.loc[graph_type_lookup["name"]==color]\n        \n    if not ax:\n        fig = plt.figure()\n        ax = plt.subplot(111)\n\n    \n    x = [int(a) for a in df_views_by_year["observation_year"]]\n    y = [int(a) for a in df_views_by_year["views"]]\n    max_y = 1.2 * max(y)\n\n    ax.scatter(x, y, marker=\'x\', s=70, color=my_color_lookup["color"])\n\n    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: \'{0:,.1f}\'.format(y/(1000*1000.0))))\n    ax.set_ylabel("views (millions)")\n\n    ax.set_xlim(min(year_range), max_year+1)\n#         ax.set_ylim(0, max_y)\n    ax.set_xlabel(\'view year\')\n#     title = plt.suptitle("Estimated views by access year, by OA type")\n#     title.set_position([.5, 1.05])\n    return df_views_by_year')


# In[9]:



# do calculations

def get_papers_by_availability_year(graph_type="closed", availability_year=2000, just_this_year=False):
    my_return = pd.DataFrame()
        
    if just_this_year:
        if graph_type == "closed":
            rows_published_this_year = articles_by_color_by_year.loc[articles_by_color_by_year["published_year"] == availability_year]
            total_this_year = rows_published_this_year.num_articles.sum()
            
            open_this_year = 0
            for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze"]:
                temp_papers = get_papers_by_availability_year(prep_graph_type, availability_year, just_this_year=False)
                temp_papers = temp_papers.loc[temp_papers.article_years_from_availability == 0]
                num_articles = temp_papers.num_articles.sum()
#                 print prep_graph_type, num_articles
                open_this_year += num_articles
            num_closed = total_this_year - open_this_year
            
            my_return = pd.DataFrame({
                "article_years_from_availability": [0],
                "num_articles": [num_closed]
            })
        else:
            prev_year_history = get_papers_by_availability_year(graph_type, availability_year-1, just_this_year=False)
            prev_year_history["article_years_from_availability"] += 1
            this_year_history = get_papers_by_availability_year(graph_type, availability_year, just_this_year=False)
            df_merged = this_year_history.merge(prev_year_history, on="article_years_from_availability", how="left")
            df_merged = df_merged.fillna(0)
            df_merged["num_articles"] = df_merged["num_articles_x"] - df_merged["num_articles_y"]
            df_merged["num_articles"][df_merged["num_articles"] < 25] = 0
            df_merged = df_merged.loc[df_merged["article_years_from_availability"] <= 10]
            my_return = pd.DataFrame({
                "article_years_from_availability": df_merged["article_years_from_availability"],
                "num_articles": df_merged["num_articles"]
            })

    else:
            
        if graph_type == "delayed_bronze":
            temp_papers = delayed_bronze_after_embargos_age_years.loc[delayed_bronze_after_embargos_age_years["prediction_year"]==availability_year]
            my_return = pd.DataFrame({
                "article_years_from_availability": temp_papers["article_age_years"],
                "num_articles": temp_papers["num_articles"]
            })

        elif graph_type == "green":

            my_green_oa = green_oa_with_dates_by_availability

            my_green_oa = my_green_oa.loc[my_green_oa["months_old_at_first_deposit"] >= -24]
            my_green_oa = my_green_oa.loc[my_green_oa["months_old_at_first_deposit"] <= 12*25]
            my_green_oa = my_green_oa.loc[my_green_oa["year_of_first_availability"] <= availability_year]

            my_green_oa_pivot = my_green_oa.pivot_table(
                         index='published_year', values=['num_articles'], aggfunc=np.sum)
            my_green_oa_pivot.reset_index(inplace=True)
            my_green_oa_pivot = my_green_oa_pivot.sort_values(by=["published_year"], ascending=False)
            my_green_oa_pivot["article_years_from_availability"] = [(availability_year - a) for a in my_green_oa_pivot["published_year"]]
            my_return = pd.DataFrame({
                "article_years_from_availability": my_green_oa_pivot["article_years_from_availability"],
                "num_articles": my_green_oa_pivot["num_articles"]
            })

        elif graph_type == "closed":
            my_return = pd.DataFrame()
            for i, year in enumerate(range(availability_year+1, 1990, -1)):
                closed_rows = get_papers_by_availability_year(graph_type, availability_year - i, just_this_year=True)
                closed_rows["article_years_from_availability"] = i
                my_return = my_return.append(closed_rows)
            
        elif graph_type == "immediate_bronze":
            temp_papers = articles_by_color_by_year_with_embargos.loc[(articles_by_color_by_year_with_embargos.oa_status=="bronze") &
                                                                   (articles_by_color_by_year_with_embargos["embargo"].isnull()) &
                                                                  (articles_by_color_by_year_with_embargos.published_year <= availability_year)]
#             temp_papers["article_years_from_availability"] = availability_year - temp_papers["published_year"]        
            temp_pivot = temp_papers.pivot_table(
                         index='published_year', values=['num_articles'], aggfunc=np.sum)
            temp_pivot.reset_index(inplace=True)
            my_return = pd.DataFrame({
                "article_years_from_availability": availability_year - temp_pivot.published_year,
                "num_articles": temp_pivot.num_articles
            })

        elif graph_type == "biorxiv": 
            my_return = biorxiv_growth_otherwise_closed.copy()
            my_return = my_return.loc[my_return["published_year"] <= availability_year]
            my_return["article_years_from_availability"] = availability_year - my_return["published_year"]
        else:
            temp_papers = articles_by_color_by_year.loc[(articles_by_color_by_year.oa_status==graph_type) &
                                                    (articles_by_color_by_year.published_year <= availability_year)]
            my_return = pd.DataFrame({
                "article_years_from_availability": [availability_year - a for a in temp_papers["published_year"]],
                "num_articles": temp_papers["num_articles"]
            })


    if not my_return.empty:
        my_return = pd.DataFrame(my_return, columns=["article_years_from_availability", "num_articles"])        
        my_return = my_return.sort_values(by="article_years_from_availability")

    return my_return




# In[10]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\ndef get_predicted_views_by_pubdate(graph_type, observation_year):\n\n    views_per_article = get_views_per_article(graph_type)\n           \n    df_views_by_year = pd.DataFrame()\n    all_papers_per_year = get_papers_by_availability_year_including_future(graph_type, observation_year, observation_year+1)\n    papers_per_year = all_papers_per_year.loc[all_papers_per_year["prediction_year"] == observation_year]\n    \n    try:\n        data_merged_clean = papers_per_year.merge(views_per_article, left_on=["article_years_from_availability"], right_on=["article_age_years"])\n        data_merged_clean = data_merged_clean.sort_values("article_age_years")\n#         print data_merged_clean.head()\n        data_merged_clean["views"] = data_merged_clean["views_per_article"] * data_merged_clean["num_articles"]\n        data_merged_clean["observation_year"] = observation_year\n        data_merged_clean["publication_year"] = observation_year - data_merged_clean["article_age_years"]\n        new_data = pd.DataFrame(data_merged_clean, columns=["publication_year", "views", "article_age_years", "observation_year"])\n        df_views_by_year = df_views_by_year.append(new_data)\n    except (ValueError, KeyError):  # happens when the year is blank\n        pass\n    \n    return df_views_by_year')


# In[ ]:





# In[11]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\ndef get_predicted_views(graph_type, now_delta_years=0):\n#     calc_min_year = 1951\n    calc_min_year = 1995\n    display_min_year = 2010\n    now_year = 2020 - now_delta_years\n    max_year = 2024\n    exponential = False\n\n    if graph_type == "biorxiv":\n        exponential = True\n        \n    views_per_article = get_views_per_article(graph_type)\n           \n    df_views_by_year = pd.DataFrame()\n    \n    all_papers_per_year = all_predicted_papers_future\n    for prediction_year in range(calc_min_year, max_year+1):        \n#     for prediction_year in range(calc_min_year, 2019):        \n#     for prediction_year in range(2017, 2019):        \n        papers_per_year = all_papers_per_year.loc[all_papers_per_year["prediction_year"] == prediction_year]\n        papers_per_year = papers_per_year.loc[papers_per_year["graph_type"] == graph_type]\n#         print views_per_article.head()\n        try:\n            data_merged_clean = papers_per_year.merge(views_per_article, left_on=["article_years_from_availability"], right_on=["article_age_years"])\n            data_merged_clean = data_merged_clean.sort_values("article_age_years")\n            win = data_merged_clean["views_per_article"] \n            sig = data_merged_clean["num_articles"]\n            views_by_access_year = signal.convolve(win, sig, mode=\'same\', method="direct")\n            y = max(views_by_access_year)\n            df_views_by_year = df_views_by_year.append(pd.DataFrame({"observation_year":[prediction_year], "views": [y]}))\n        except (ValueError, KeyError):  # happens when the year is blank\n            pass\n        \n\n    return df_views_by_year')


# In[12]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\ndef get_papers_by_availability_year_total(availability_year):\n    all_data = pd.DataFrame()\n    for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze", "closed"]:\n#     for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze"]:\n        temp_papers = get_papers_by_availability_year_including_future(prep_graph_type, availability_year, availability_year+1)\n        temp_papers["graph_type"] = prep_graph_type\n#         print prep_graph_type\n#         print "{:,.0f}".format(temp_papers.num_articles.max()), "{:,.0f}".format(temp_papers.num_articles.sum())\n#         print "\\n"\n        all_data = all_data.append(temp_papers)\n    return all_data\n\ndef get_views_per_year_total():\n    all_data = pd.DataFrame()\n    for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze", "closed"]:\n        temp_papers = get_views_per_year(prep_graph_type)\n        temp_papers["graph_type"] = prep_graph_type\n#         print prep_graph_type\n#         print "{:,.0f}".format(temp_papers.num_views_per_year.max()), "{:,.0f}".format(temp_papers.num_views_per_year.sum())\n#         print "\\n"\n        all_data = all_data.append(temp_papers)\n    return all_data\n\n\n\ndef get_views_per_article_total():\n    all_data = pd.DataFrame()\n    for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze", "closed"]:\n        temp_papers = get_views_per_article(prep_graph_type)\n#         print prep_graph_type\n#         print "{:,.0f}".format(temp_papers.views_per_article.max()), "{:,.0f}".format(temp_papers.views_per_article.sum())\n#         print "\\n"\n        temp_papers["graph_type"] = prep_graph_type\n        all_data = all_data.append(temp_papers)\n    return all_data\n\n\ndef get_predicted_views_total(observation_year):\n    all_data = pd.DataFrame()\n    for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze", "closed"]:\n#     for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze"]:\n        temp_papers = get_predicted_views(prep_graph_type, observation_year)\n        temp_papers["graph_type"] = prep_graph_type\n#         print prep_graph_type        \n        all_data = all_data.append(temp_papers)\n    return all_data\n\ndef get_predicted_views_by_pubdate_total(observation_year):\n    all_data = pd.DataFrame()\n#     for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze"]:\n    for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze", "closed"]:\n        temp_papers = get_predicted_views_by_pubdate(prep_graph_type, observation_year)\n        temp_papers["graph_type"] = prep_graph_type\n#         print prep_graph_type\n        all_data = all_data.append(temp_papers)\n    return all_data')


# In[13]:


def get_views_per_year(graph_type):
    if graph_type == "delayed_bronze":
        views_per_year = views_by_age_years.loc[(views_by_age_years.oa_status=="bronze") &
                                                       (views_by_age_years.delayed_or_immediate=="delayed")]
    elif graph_type == "immediate_bronze":
        views_per_year = views_by_age_years.loc[(views_by_age_years.oa_status=="bronze") &
                                                       (views_by_age_years["delayed_or_immediate"]=="immediate")]

    else:
        views_per_year = views_by_age_years.loc[(views_by_age_years.oa_status==graph_type)]

    views_per_year["num_views_one_month"] = views_per_year["num_views"]  # this is just for one month
    views_per_year["num_views_per_year"] = 12.0 * views_per_year["num_views_one_month"]
    del views_per_year["num_views"]
    del views_per_year["delayed_or_immediate"]
    views_per_year = views_per_year.sort_values(by="article_age_years")
    views_per_year = views_per_year.loc[views_per_year["article_age_years"] < 15]
    
    return views_per_year     


def get_views_per_article(graph_type):
    if graph_type == "biorxiv":
        graph_type = "green"
        
    views_per_year = get_views_per_year(graph_type)
    papers_per_year = get_papers_by_availability_year(graph_type, 2018, just_this_year=False)
    papers_per_year["article_age_years"] = papers_per_year["article_years_from_availability"]
    papers_per_year = papers_per_year.loc[(papers_per_year["article_age_years"] <=15 )]

    data_merged_clean = papers_per_year.merge(views_per_year, on=["article_age_years"])        
    data_merged_clean["views_per_article"] = data_merged_clean["num_views_per_year"] / data_merged_clean["num_articles"]

    views_per_article = pd.DataFrame(data_merged_clean, columns=["article_age_years", "views_per_article"])
    views_per_article = views_per_article.sort_values(by="article_age_years")

    if graph_type=="delayed_bronze":
        # otherwise first one is too high because number articles too low in year 0 for delayed subset
        views_per_article.loc[views_per_article.article_age_years==0, ["views_per_article"]] = float(views_per_article.loc[views_per_article.article_age_years==1].views_per_article)

    return views_per_article


# In[14]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\ndef plot_area_and_proportion(df, color_type, start_year, end_year, divide_year, \n                             xlabel="year of publication",\n                             fancy=None):\n    if color_type=="simple":\n        my_colors = oa_status_colors\n        my_color_order = oa_status_order\n        color_column = "color"\n    elif color_type=="standard":\n        my_colors = graph_type_colors\n        my_color_order = graph_type_order\n        color_column = "graph_type"\n    else:\n        my_colors = graph_type_colors_plus_biorxiv\n        my_color_order = graph_type_order_plus_biorxiv\n        color_column = "graph_type"\n        \n    all_data_pivot = df.pivot_table(\n                 index=\'x\', columns=color_column, values=[\'y\'], aggfunc=np.sum)\\\n           .sort_index(axis=1, level=1)\\\n           .swaplevel(0, 1, axis=1)\n    all_data_pivot.columns = all_data_pivot.columns.levels[0]\n\n    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=False)\n    plt.tight_layout(pad=0, w_pad=2, h_pad=1)\n    plt.subplots_adjust(hspace=1)\n\n    all_data_pivot_graph = all_data_pivot\n    ylabel = "articles (millions)"\n    if fancy=="cumulative":\n        ylabel = "cumulative articles (millions)"\n        all_data_pivot_graph = all_data_pivot_graph.cumsum(0)\n    elif fancy=="diff":\n        ylabel = "newly available articles (millions)"\n        all_data_pivot_graph = all_data_pivot_graph.diff()\n    all_data_pivot_graph = all_data_pivot_graph.loc[all_data_pivot_graph.index > 1950]\n    all_data_pivot_graph = all_data_pivot_graph.loc[all_data_pivot_graph.index <= end_year]\n        \n    # print all_data_pivot_graph\n    all_data_pivot_actual = all_data_pivot_graph.loc[all_data_pivot_graph.index <= divide_year+1]\n    my_plot = all_data_pivot_actual[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax1)\n    if end_year > divide_year:\n        all_data_pivot_projected = all_data_pivot_graph.loc[all_data_pivot_graph.index > divide_year]\n        my_plot = all_data_pivot_projected[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax1, alpha=0.6)\n    ax1.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(\'{x:.0f}\'))\n    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: \'{0:,.0f}\'.format(y/(1000*1000.0))))\n    ax1.set_xlabel(xlabel)\n    ax1.set_ylabel(ylabel)    \n    ax1.set_xlim(start_year, end_year)\n    ax1.set_ylim(0, 1.2*max(all_data_pivot_graph.sum(1)))\n#     ax1.set_title("Number of papers");\n    handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:len(my_colors)]), reversed(labels[0:len(my_colors)]), loc=\'upper left\');  # reverse to keep order consistent\n\n    df_diff_proportional = all_data_pivot_graph.div(all_data_pivot_graph.sum(1), axis=0)\n    all_data_pivot_actual = df_diff_proportional.loc[all_data_pivot_graph.index <= divide_year+1]\n    my_plot = all_data_pivot_actual[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax2)\n    if end_year > divide_year:\n        all_data_pivot_projected = df_diff_proportional.loc[all_data_pivot_graph.index > divide_year]\n        my_plot = all_data_pivot_projected[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax2, alpha=0.6)\n    my_plot.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))\n    ax2.set_xlabel(xlabel)\n    ax2.set_ylabel(\'proportion of articles\')\n#     ax2.set_title("Proportion of papers");\n    ax2.set_xlim(start_year, end_year)\n    ax2.set_ylim(0, 1)    \n    handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:len(my_colors)]), reversed(labels[0:len(my_colors)]), loc=\'upper left\');  # reverse to keep order consistent\n\n    plt.tight_layout(pad=.5, w_pad=4, h_pad=2.0) \n    return (all_data_pivot_graph, df_diff_proportional)')


# In[15]:


# plot graphs duplicate new

def get_long_data(graph_type):
    full_range = range(1990, 2020)

    totals_bronze = pd.DataFrame()
    for i, prediction_year in enumerate(full_range):
        new_frame = get_papers_by_availability_year(graph_type, prediction_year, just_this_year=True)
        new_frame["prediction_year"] = prediction_year
        new_frame["published_year"] = [int(prediction_year - a) for a in new_frame["article_years_from_availability"]]
        totals_bronze = totals_bronze.append(new_frame)

    long_data_for_plot = totals_bronze
    long_data_for_plot = long_data_for_plot.loc[long_data_for_plot["article_years_from_availability"] < 15]
    return long_data_for_plot

def first_detailed_plots(graph_type):
    my_color_lookup = graph_type_lookup.loc[graph_type_lookup["name"]==graph_type]    

    long_data_for_plot = get_long_data(graph_type)
    pivot_data_for_plot = long_data_for_plot.pivot_table(
                 index='published_year', columns='prediction_year', values=['num_articles'], aggfunc=np.sum)\
           .sort_index(axis=1, level=1)\
           .swaplevel(0, 1, axis=1)
    pivot_data_for_plot.columns = pivot_data_for_plot.columns.levels[0]
    pivot_data_for_plot[pivot_data_for_plot < 0] = 0
    pivot_data_for_plot["published_year"] = [int(a) for a in pivot_data_for_plot.index]

    years = range(2015, 2018+1)

    historical_graphs = False
    color_idx = np.linspace(0, 1, len(years))
    fig, axes = plt.subplots(1, len(years), figsize=(12, 3), sharex=True, sharey=True)
    axes_flatten = axes.flatten()
    axis_index = 0
    max_y_for_this_plot = max(pivot_data_for_plot.max(1))

    for i, prediction_year in enumerate(years):
        ax = axes_flatten[axis_index]        
        axis_index += 1
        rows = pivot_data_for_plot.copy()
        rows = rows.loc[pd.notnull(rows[prediction_year])]
        x = [int(a) for a in rows.index]
        y = [int(a) for a in rows[prediction_year]]
        ax.bar(x, y, color=my_color_lookup["color"])
        ax.set_ylim(0, 1.2*max_y_for_this_plot)    
        ax.set_xlim(2010, 2019)
        if ax.get_legend():
            ax.get_legend().remove()  
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)   
        ax.set_xlabel("year of publication")
        ax.set_title("year first available OA\n{}".format(prediction_year))

    axes_flatten[0].set_ylabel("articles\nfirst made available")        
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.subplots_adjust(hspace=0)
    plt.show()
        
        


# In[ ]:





# In[16]:


def make_detailed_plots(graph_type):
    num_subplots = 8

    long_data_for_plot = get_long_data(graph_type)
    pivot_data_for_plot = long_data_for_plot.pivot_table(
                 index='published_year', columns='prediction_year', values=['num_articles'], aggfunc=np.sum)\
           .sort_index(axis=1, level=1)\
           .swaplevel(0, 1, axis=1)
    pivot_data_for_plot.columns = pivot_data_for_plot.columns.levels[0]
    pivot_data_for_plot[pivot_data_for_plot < 0] = 0
    # print pivot_data_for_plot

    years = [year for year in pivot_data_for_plot.columns if year > 1990]

    for historical_graphs in (False, True):
        color_idx = np.linspace(0, 1, len(years))
        fig, axes = plt.subplots(len(years[-num_subplots:]), 1, figsize=(7, 6), sharex=True, sharey=True)
        axes_flatten = axes.flatten()
        axis_index = 0
        max_y_for_this_plot = max(pivot_data_for_plot.max(1))
        for i, prediction_year in zip(color_idx[-num_subplots:], years[-num_subplots:]):
            ax = axes_flatten[axis_index]        
            axis_index += 1
            if historical_graphs:
                pivot_data_for_plot[range(2000, prediction_year+1)].plot.area(stacked=True,  alpha=0.4, ax=ax, color=[plt.cm.jet(i) for x in range(2000, prediction_year)])
                try:
                    pivot_data_for_plot[range(2000, prediction_year)].plot.area(stacked=True,  ax=ax, alpha=.9, color="lightgray")
                    ax.set_ylim(0, 3*max_y_for_this_plot)    
                except TypeError:
                    pass       
            else:
                pivot_data_for_plot[prediction_year].plot.area(stacked=False, ax=ax,  alpha=.4, color=plt.cm.jet(i))
                ax.set_ylim(0, 1.2*max_y_for_this_plot)    
            ax.set_xlim(2009, 2018)
            if ax.get_legend():
                ax.get_legend().remove()        
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)   
            y_label = "{} made available during {}:".format(graph_type, prediction_year) 
            ax.set_ylabel(y_label, rotation='horizontal', labelpad=150, verticalalignment="center")
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 3))
    pivot_data_for_plot[years].plot.area(stacked=True, ax=ax1,   alpha=.4, cmap=plt.cm.jet)
    ax1.set_xlim(2000, 2018)
    legend_handles, legend_labels = ax1.get_legend_handles_labels(); ax1.legend(reversed(legend_handles[-8:]), reversed(legend_labels[-8:]), loc='upper left');  # reverse to keep order consistent
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
    ax1.axvline(x=2015, color='black')
    ax1.set_title("Total {} OA available in 2019, by year of availability and publication year".format(graph_type));
    ax1.set_ylabel("number of articles")
    ax1.set_xlabel("published year")
    
    plt.tight_layout()
    plt.show()


# In[17]:



def make_zoom_in_plot(graph_type):
    full_range = range(1990, 2020)
    long_data_for_plot = get_long_data(graph_type)
    color_idx = np.linspace(0, 1, len(full_range))

    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4))
    data_for_this_plot = long_data_for_plot
    data_for_this_plot = data_for_this_plot.loc[data_for_this_plot["published_year"]==2015]
    total_sum = data_for_this_plot["num_articles"].sum()
    data_for_this_plot = data_for_this_plot.loc[data_for_this_plot["num_articles"]/total_sum>=0.01]
#     print data_for_this_plot
    # data_for_this_plot = data_for_this_plot.drop(columns=["article_age_months"])
    pivot_df = data_for_this_plot.pivot_table(index='published_year', columns='prediction_year', aggfunc=np.sum)
    pivot_df = pivot_df.div(pivot_df.sum(1), axis=0)
    pivot_df.plot.bar(stacked=True, alpha=.4, ax=ax1, colors=[plt.cm.jet(a) for a in list(color_idx[-len(pivot_df.sum(0)):])])
    ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    plt.ylabel('proportion of articles')
    plt.title("Proportion of {} articles published in 2015".format(graph_type));
    ax1.set_xlabel("")
    ax1.set_xticks([]) 
    legend_handles, legend_labels = ax1.get_legend_handles_labels(); 
    cleaned_legend_labels = [a[-5:-1] for a in legend_labels]
    legend_length = len(data_for_this_plot)  # just the nonzero ones
    ax1.legend(reversed(legend_handles[-legend_length:]), reversed(cleaned_legend_labels[-legend_length:]), loc='upper left');  # reverse to keep order consistent


# In[18]:


# Nonlinear curve fit with confidence interval
def curve_fit_with_ci(graph_type, papers_per_year_historical, curve_type, ax=None):
    my_rows = papers_per_year_historical.loc[papers_per_year_historical.article_years_from_availability <= 5]
    my_rows = my_rows.loc[my_rows.prediction_year >= 2000]
    my_rows = my_rows.loc[my_rows.prediction_year < 2018]
    x = my_rows.groupby("prediction_year", as_index=False).sum().prediction_year
    y = my_rows.groupby("prediction_year", as_index=False).sum().num_articles

    my_color_lookup = graph_type_plus_biorxiv_lookup.loc[graph_type_plus_biorxiv_lookup["name"]==graph_type]
    my_color = my_color_lookup.iloc[0]["color"]
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=False)
    ax.plot(x, y, 'o', color=my_color)
    ax.set_xlim(2000, 2025)
    ax.set_ylabel("articles (millions)")
    ax.set_title("{}".format(graph_type))
    
    if curve_type == "no_line":
        ax.set_xlabel("year of observation")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.2f}'.format(y/(1000*1000.0))))        
        return
    
    if curve_type == "linear":
        initial_guess=None
        def func(x, a, b):
            return a * (x - 2000) + b
    elif curve_type == "exp":
        if graph_type == "biorxiv":
            initial_guess=(5, 1, 1)
            def func(x, a, b, d):
               return b + a * np.exp((x - 2014)/d)
        else:
            initial_guess=(14287, 21932, 5)
            def func(x, a, b, d):
               return b + a * np.exp((x - 2000)/d)
    elif curve_type == "negative_exp":
        initial_guess=(1731700, 22962997, -7)
        def func(x, a, b, d):
           return b - a * np.exp((x - 2000)/d) 

    pars, pcov = curve_fit(func, x, y, initial_guess)

    xfit_extrap = range(2000, 2040+1)
    if curve_type == "linear":
        yfit_extrap = [func(a, pars[0], pars[1]) for a in xfit_extrap]
        yfit = [func(a, pars[0], pars[1]) for a in x]
    else:
        yfit_extrap = [func(a, pars[0], pars[1], pars[2]) for a in xfit_extrap]
        yfit = [func(a, pars[0], pars[1], pars[2]) for a in x]
        
    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(y)    # number of data points
    p = len(pars) # number of parameters
    dof = max(0, n - p) # number of degrees of freedom
    tval = t.ppf(1.0-alpha/2., dof) # student-t value 

    residuals = y - yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    fit_string = ""
    for i, p, var in zip(range(n), pars, np.diag(pcov)):
        sigma = var**0.5
        fit_string += ' p{}: {} [{}  {}] '.format(i, 
                                            round(p, 3),
                                            round(p - sigma*tval, 3),
                                            round(p + sigma*tval, 3))
    fit_string += "{}".format(round(r_squared, 3))
#     print "{} {} {}".format(graph_type, curve_type, fit_string)

    ax.plot(xfit_extrap[0:25], yfit_extrap[0:25], '-', color=my_color)
    ax.set_xlabel("r^2={}".format(round(r_squared, 3)))
    if max(yfit_extrap) > 100000:
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.2f}'.format(y/(1000*1000.0))))
    my_return = pd.DataFrame({
        "x": xfit_extrap,
        "y": yfit_extrap,
        "r_squared": [r_squared for y in yfit_extrap]
    })
    return my_return


# #### Code: SQL

# See notebook.

# In[19]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# query for articles_by_color_by_year_with_embargos and articles_by_color_by_year\n\nq = """\nselect date_part(\'year\', fixed.published_date)::int as published_year, \nfixed.oa_status,\ndelayed.embargo,\ncount(*) as num_articles\nfrom unpaywall u\nleft join journal_delayed_oa_active delayed on u.journal_issn_l = delayed.issn_l\njoin unpaywall_updates_view fixed on fixed.doi=u.doi\nwhere genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\nand published_year > \'1950-01-01\'::timestamp\ngroup by published_year, fixed.oa_status, embargo\norder by published_year asc\n"""\narticles_by_color_by_year_with_embargos = read_from_file_or_db("articles_by_color_by_year_with_embargos", q)\n\narticles_by_color_by_year = articles_by_color_by_year_with_embargos.drop(columns = ["embargo"])\narticles_by_color_by_year = articles_by_color_by_year.groupby([\'published_year\', \'oa_status\']).sum()\narticles_by_color_by_year.reset_index(inplace=True)')


# In[20]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# query for articles_by_graph_type_by_year\n\nq = """\nselect date_part(\'year\', fixed.published_date) as published_year, \nfixed.oa_status,\ncase when fixed.oa_status=\'bronze\' and delayed.embargo is not null then \'delayed_bronze\' \n    when fixed.oa_status=\'bronze\' and delayed.embargo is null then \'immediate_bronze\' \n    else fixed.oa_status end\n    as graph_type,\ncount(*) as num_articles\nfrom unpaywall u\nleft join journal_delayed_oa_active delayed on u.journal_issn_l = delayed.issn_l\njoin unpaywall_updates_view fixed on fixed.doi=u.doi\nwhere genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\nand published_year > \'1950-01-01\'::timestamp\nand published_year < \'2019-01-01\'::timestamp\ngroup by published_year, fixed.oa_status, graph_type\norder by published_year asc\n"""\narticles_by_graph_type_by_year = read_from_file_or_db("articles_by_graph_type_by_year", q)')


# In[21]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# query for views_by_age_months_no_color_full_year.  maybe don\'t need this one in the final paper?\n\nq = """\nselect datediff(\'days\', fixed.published_date, received_at_raw::timestamp)/30 as article_age_months, \ncount(u.doi) as num_views \nfrom papertrail_unpaywall_extracted extracted \njoin unpaywall u on extracted.doi=u.doi \njoin unpaywall_updates_view fixed on fixed.doi=u.doi\nwhere genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\nand fixed.published_date > \'1950-01-01\'::timestamp\nand extracted.doi not in (\'10.1038/nature21360\', \'10.1038/nature11723\')\ngroup by article_age_months\norder by article_age_months asc\n\n"""\nviews_by_age_months_no_color_full_year = read_from_file_or_db("views_by_age_months_no_color_full_year", q)')


# In[22]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# query for views_by_age_months\n# not used by analysis but here for data dump\n\nq = """\nselect datediff(\'days\', fixed.published_date, received_at_raw::timestamp)/30 as article_age_months, \nfixed.oa_status,\ncount(u.doi) as num_views \nfrom papertrail_unpaywall_extracted extracted\njoin unpaywall u on extracted.doi=u.doi \njoin unpaywall_updates_view fixed on fixed.doi=u.doi\nwhere genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\nand fixed.published_date > \'1950-01-01\'::timestamp\nand fixed.published_date < current_date\nand received_at_raw > \'2019-07-01\'\nand received_at_raw <= \'2019-08-01\'\nand extracted.doi != \'10.1038/nature21360\'\ngroup by article_age_months, fixed.oa_status\norder by article_age_months asc\n"""\nviews_by_age_months = read_from_file_or_db("views_by_age_months", q)\n')


# In[23]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# query for views_by_age_years\n\nq = """\nselect datediff(\'days\', fixed.published_date, received_at_raw::timestamp)/(30*12) as article_age_years, \nfixed.oa_status,\ncase when fixed.oa_status=\'bronze\' and journal_issn_l in (select issn_l from journal_delayed_oa_active) then \'delayed\' when fixed.oa_status=\'bronze\' then \'immediate\' else null end as delayed_or_immediate,\ncount(u.doi) as num_views \nfrom papertrail_unpaywall_extracted extracted \njoin unpaywall u on extracted.doi=u.doi \njoin unpaywall_updates_view fixed on fixed.doi=u.doi\nwhere genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\nand fixed.published_date > \'1950-01-01\'::timestamp\nand fixed.published_date < current_date\nand received_at_raw > \'2019-07-01\'\nand received_at_raw <= \'2019-08-01\'\nand extracted.doi != \'10.1038/nature21360\'\ngroup by article_age_years, fixed.oa_status, delayed_or_immediate\norder by article_age_years asc\n"""\nviews_by_age_years = read_from_file_or_db("views_by_age_years", q)')


# In[24]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\nq = """\nselect date_part(\'year\', min_record_timestamp) as year_of_first_availability, \ndatediff(\'days\', fixed.published_date, min_record_timestamp)/30 as months_old_at_first_deposit,\ndate_part(\'year\', fixed.published_date) as published_year,\ncount(*) as num_articles\nfrom unpaywall u\njoin unpaywall_pmh_record_min_timestamp pmh on u.doi=pmh.doi\njoin unpaywall_updates_view fixed on fixed.doi=u.doi\nwhere fixed.oa_status = \'green\'\nand genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\nand year_of_first_availability is not null\ngroup by year_of_first_availability, months_old_at_first_deposit, published_year\n"""\ngreen_oa_with_dates_by_availability = read_from_file_or_db("green_oa_with_dates_by_availability", q)')


# In[25]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# queries delayed_bronze_after_embargos_age_months\n# not used by analysis but here for data dump\n\nmin_prediction_year = 1949\nmax_prediction_year = 2019 + 1\nprediction_year_range = range(min_prediction_year, max_prediction_year)\ndelayed_bronze_after_embargos_age_months = pd.DataFrame()\n\nfor i, prediction_year in enumerate(range(min_prediction_year - 1, max_prediction_year)):\n    \n    q = """\n    select \n    datediff(\'days\', fixed.published_date, \'{prediction_year}-01-01\'::timestamp)/30 as article_age_months, \n    --datediff(\'days\', fixed.published_date, current_date)/30 as article_age_months_from_now, \n    {prediction_year} as prediction_year,\n    count(*) as num_articles\n    from unpaywall u\n    left join journal_delayed_oa_active delayed on u.journal_issn_l = delayed.issn_l\n    join unpaywall_updates_view fixed on fixed.doi=u.doi\n    where genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\n    and fixed.oa_status = \'bronze\'\n    and delayed.embargo is not null\n    and fixed.published_date > \'1950-01-01\'::timestamp\n    and fixed.published_date <= ADD_MONTHS(\'{prediction_year}-01-01\'::timestamp, -embargo::integer)\n    group by prediction_year, article_age_months\n    order by prediction_year, article_age_months asc\n    """.format(prediction_year=prediction_year)\n\n    filename_root = "delayed_bronze_sql_parts/{varname}_{index}".format(varname="bronze_rows_by_month", index=i)    \n    bronze_rows = read_from_file_or_db(filename_root, q)\n    \n    delayed_bronze_after_embargos_age_months = delayed_bronze_after_embargos_age_months.append(bronze_rows)\ndelayed_bronze_after_embargos_age_months.to_csv("data/delayed_bronze_after_embargos_age_months.csv")\n')


# In[26]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\n# queries delayed_bronze_after_embargos_age_years\n\nmin_prediction_year = 1949\nmax_prediction_year = 2019 + 1\nprediction_year_range = range(min_prediction_year, max_prediction_year)\ndelayed_bronze_after_embargos_age_years = pd.DataFrame()\n\nfor i, prediction_year in enumerate(range(min_prediction_year - 1, max_prediction_year)):\n    \n    q = """  \n    select \n    datediff(\'days\', fixed.published_date, \'{prediction_year}-01-01\'::timestamp)/(30*12) as article_age_years, \n    {prediction_year} as prediction_year,\n    count(*) as num_articles\n    from unpaywall u\n    left join journal_delayed_oa_active delayed on u.journal_issn_l = delayed.issn_l\n    join unpaywall_updates_view fixed on fixed.doi=u.doi\n    where genre = \'journal-article\' and journal_issn_l not in (\'0849-6757\', \'0931-7597\')\n    and fixed.oa_status = \'bronze\'\n    and delayed.embargo is not null\n    and fixed.published_date > \'1950-01-01\'::timestamp\n    and fixed.published_date <= ADD_MONTHS(\'{prediction_year}-01-01\'::timestamp, -embargo::integer)\n    \n    group by prediction_year, article_age_years\n    order by prediction_year, article_age_years asc\n    """.format(prediction_year=prediction_year)\n\n    filename_root = "delayed_bronze_sql_parts/{varname}_{index}".format(varname="bronze_rows_by_year", index=i)\n    bronze_rows_by_year = read_from_file_or_db(filename_root, q)\n    \n    delayed_bronze_after_embargos_age_years = delayed_bronze_after_embargos_age_years.append(bronze_rows_by_year)\ndelayed_bronze_after_embargos_age_years.to_csv("data/delayed_bronze_after_embargos_age_years.csv")')


# In[27]:


get_ipython().run_cell_magic(u'capture', u'--no-stderr --no-stdout --no-display', u'\nq = """select u.year::numeric as published_year, count(distinct u.doi) as num_articles \nfrom unpaywall u\njoin unpaywall u_biorxiv_record on u_biorxiv_record.doi = replace(u.best_url, \'https://doi.org/\', \'\')\nwhere u.doi not like \'10.1101/%\' and u.best_url like \'%10.1101/%\'\nand datediff(\'days\', u_biorxiv_record.published_date::timestamp, u.published_date::timestamp)/(30.0) >= 0\nand u.year >= 2013 and u.year < 2019\ngroup by u.year\norder by u.year desc\n"""\nbiorxiv_growth_otherwise_closed = read_from_file_or_db("biorxiv_growth_otherwise_closed", q)')


# *---- delete the text to the line above in the final paper ----*

# <a id="section-4"></a>
# ## 4. Methods and Results

# <a id="section-4-1"></a>
# ### 4.1 Past OA Publication, by date of observation

# <a id="section-4-1-1"></a>
# #### 4.1.1 OA lag
# 
# For Gold OA and Hybrid OA understanding OA lag is easy -- there is no lag: papers become OA at the time of publication. 
# 
# For Green and Bronze OA the lag is more complicated. Authors often self-archive (upload their paper to a repository) months or years after the official publication date of the paper, typically  because the journal has a policy that authors must wait a certain length of time (the "embargo period") before self-archiving. Funder policies that mandate Green OA often allow a delay between publication and availability (notably the National Institutes of Health in the USA allows a 12 month embargo, which is relevant for most of the content in the large PMC repository). Finally, some journals open up their back catalogs once articles reach a certain age, which has been called "delayed OA" (Laakso and Björk, 2013) and we consider an important subset of Bronze.
# 
# We explore and model these dynamics below.

# <a id="section-4-1-2"></a>
# #### 4.1.2. OA lag for Green OA

# In[28]:


register_new_figure("oa_lag_green");


# Calculating OA lag requires data on both when an article was first published in its journal and the date it was first made OA.  
# 
# The date an article becomes Green OA can be derived from the date it was made available in a repository, which we can get from its matched [OAI-PMH records](https://www.openarchives.org/pmh/) (as harvested by Unpaywall).  
# 
# {{ print figure_link("oa_lag_green") }} shows four plots: the leftmost plot shows Green OA articles that were first made OA in 2015, the second plot shows Green OA articles that were first made OA in 2016, and so on.  Each plot is a histogram of number of articles by date of publication.  
# 

# In[29]:


first_detailed_plots("green")


# **{{print figure_link("oa_lag_green")}}: OA lag for Green OA.**  Each plot shows articles that were first made available during the given year of observation, by year of their publication on the x-axis.

# By looking at the first plot in depth, we can see that a few articles are made available *before* they are actually published (articles published in 2016 or 2017) -- these were  preprints, submitted before publication.  Continuing to look at the first row, we can see the bulk of the articles that became available in 2015 were published in 2015 (lag of zero years) or in 2014 (lag of 1 year).  A few were published in 2013 (an OA lag of 2 years), and then a long tail represents the backfilling of older articles.  
# 
# Looking now at all plots in {{ print figure_link("oa_lag_green") }}, we can see that a similar OA lag pattern (a few preprints are available before publication, most articles become available within a 3 year OA lag, then a long tail) has held for the last four years of Green OA availability (the distribution of the bars are similar in all four graphs). 

# We can also see that the relative amount of green OA is growing slightly by year of OA-first-availability (the area under the whole histogram gets higher with subsequent histograms).  Green OA appears to be growing.  We will explore this further in [Section 4.2](#section-4-2).
# 
# More details on Green OA lag are included in Supplementary Information, [Section 11.1](#section-11-1).

# <a id="section-4-1-3"></a>
# #### 4.1.3 OA lag for Bronze Delayed OA

# There was no recent, complete, publicly-available list of Delayed OA journals, so we derived a list empirically based on the Unpaywall database. We have made our list publicly available: details are in [Section 7.2](#section-7-2).
# 
# To create the list we started by looking at existing compilations of Delayed OA journals, including:
# 
# -   <https://www.elsevier.com/about/open-science/open-access/open-archive>
# 
# -   <http://highwire.stanford.edu/cgi/journalinfo#loc>
# 
# -   <https://www.ncbi.nlm.nih.gov/pmc/journals/?filter=t3&titles=current&search=journals#csvfile>
# 
# -   <https://en.wikipedia.org/wiki/Category:Delayed_open_access_journals>
# 
# - [Delayed open access: An overlooked high‐impact category of openly available scientific literature](https://helda.helsinki.fi/bitstream/10138/157658/3/Laakso_Bj_rk_2013_Delayed_OA.pdf) by Laakso and Björk (2013).
# 
# From those sources we determined that almost all embargoes for Delayed OA journals are at 6, 18, 24, 36, 48, or 60 months.  

# Next we used the Unpaywall data to calculate the OA rate of all journals, partitioned by age of their articles.  We looked at Bronze OA rates before and after each of these common month cutoffs, highlighting cutoffs where OA was much less than 90% before the cutoff and 90% or higher afterwards.  For each cutoff that looked like a Delayed OA candidate, we manually examined the full OA pattern for the journal and made a judgment call about whether it had an OA pattern consistent with a Delayed OA journal (low OA rates for articles until an embargo date, then high OA rates).  We finally cross-referenced this empirically derived list with the sources again to see if it was roughly equivalent for journals on both lists -- it is, and the empirically derived list is more comprehensive. 

# Our resulting list includes 3.6 million articles (4.9% of all articles) published in 546 journals, with the following embargo lengths:
# 
# embargo	length (months)|number of journals|number of articles
# ---|---|---
# 6	|58 | 511,326
# 12	|175| 1,608,597
# 18	|137 | 68,9820
# 24	|42 | 188,949
# 36	|71 | 269,186
# 48	|63 | 316,510
# **Total**	|**546** | **3,584,388**

# In[30]:


register_new_figure("oa_lag_delayed_bronze");


# We used this list to split articles labelled "Bronze" by Unpaywall into two categories:  "Delayed Bronze" for articles published in journals in our Delayed OA list, and "Immediate Bronze" for all others.
# 
# Immediate Bronze articles have no OA lag:  they become available on the publisher site immediately.
# 
# We estimate the OA lag for a Delayed Bronze OA article as the Delayed OA embargo for journal it is published in.  From there we can also estimate the date it first became OA by adding the embargo period to the publication date of the article.
# 
# {{ print figure_link("oa_lag_delayed_bronze") }} shows four plots: the leftmost plot shows Delayed Bronze OA articles that were first made OA in 2015, the second plot shows Delayed Bronze OA articles that were first made OA in 2016, and so on.  Each plot is a histogram of number of articles by date of publication.  
# 
# The distribution of Delayed Bronze OA articles by date first made OA is shown in {{ print figure_link("oa_lag_delayed_bronze") }}, as histograms by publication date. Most articles become available after a 1 year lag.  Bumps that represent articles that become available at 24, 36, and 48 months are also clearly visible.

# In[31]:


first_detailed_plots("delayed_bronze")


# **{{print figure_link("oa_lag_delayed_bronze")}}: OA lag for Delayed Bronze OA.**  Each plot shows articles that were first made available during the given year of observation, by year of their publication on the x-axis.

# By looking at the first plot of {{ print figure_link("oa_lag_delayed_bronze") }} in depth, we can see that most articles first made available in Delayed Bronze OA journals were made available with 1 year OA lag, in 2014.  A few were made available with a lag of less than one year, 2 years, or 4 years.  
# 
# We can also see that the relative amount of Delayed Bronze OA is not growing very much by year of OA-first-availability (the area under the whole histogram gets higher with subsequent histograms is approximately the same for all histograms).  Delayed Bronze OA is not growing quickly.  We will explore this further in [Section 4.2](#section-4-2).
# 
# More details on Delayed Bronze OA lag are included in Supplementary Information, [Section 11.2](#section-11-2).

# <a id="section-4-1-4"></a>
# #### 4.1.4 Closed access at date of observation

# We consider an article Closed if it has been published and is not considered OA at the time of observation.

# <a id="section-4-1-5"></a>
# #### 4.1.5 Past OA by date of observation and date of publication

# In[32]:


register_new_figure('small-multiples-num-papers-past');


# We combine the OA lag data above to describe OA by date of observation for all OA types, in {{ print figure_link('small-multiples-num-papers-past')}}.  
# 
# Each column is a year of observation, from 2014 to 2018.   Each row is a different OA type.  Each mini plot is a histogram of all articles available by publication date, for the given observation year and OA type.  
# 
# This figure differs from {{ print figure_link("oa_lag_green") }} and {{ print figure_link("oa_lag_delayed_bronze") }} in that it is cumulative over date of first availability:  it shows all papers published prior to the year of observation.

# In[33]:


# start here 

now_year = 2018
papers_per_year_historical = pd.DataFrame()
for graph_type in graph_type_order:
    for prediction_year in range(1990, now_year+1):        
        papers_per_year = get_papers_by_availability_year(graph_type, prediction_year, just_this_year=True)
        papers_per_year["graph_type"] = graph_type
        papers_per_year["prediction_year"] = prediction_year
        papers_per_year_historical = papers_per_year_historical.append(papers_per_year)
        
papers_per_year_historical_cumulative = pd.DataFrame()
for graph_type in graph_type_order:
    for prediction_year in range(1990, now_year+1):        
        papers_per_year = get_papers_by_availability_year(graph_type, prediction_year, just_this_year=False)
        papers_per_year["graph_type"] = graph_type
        papers_per_year["prediction_year"] = prediction_year
        papers_per_year_historical_cumulative = papers_per_year_historical_cumulative.append(papers_per_year)        


# In[34]:


my_range = range(2014, 2018+1)

fig, axes = plt.subplots(len(graph_type_order)+1, len(my_range), figsize=(12, 6), sharex=True, sharey=False)
axes_flatten = axes.flatten()
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

i = 0
for observation_year in my_range:
    ax = axes_flatten[i]
    ax.set_axis_off() 
    column_label = "observation year\n{}".format(observation_year)
    ax.text(.3, .2, column_label,
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=14,
        transform=ax.transAxes)
    i += 1

for graph_type in graph_type_order[::-1]:
    for observation_year in my_range:    
        ax = axes_flatten[i]
        this_data = papers_per_year_historical_cumulative.copy()
        this_data = this_data.loc[this_data.graph_type == graph_type]
        this_data = this_data.loc[this_data.prediction_year == observation_year]
        this_data["publication_date"] = [int(observation_year - a) for a in this_data.article_years_from_availability]
        new_data = graph_available_papers_in_observation_year_by_pubdate(graph_type, this_data, observation_year, ax=ax)

        y_max = papers_per_year_historical_cumulative.loc[(papers_per_year_historical_cumulative.graph_type == graph_type) &
                                               (papers_per_year_historical_cumulative.prediction_year <= max(my_range))]["num_articles"].max()
        ax.set_ylim(0, 1.2*y_max)
        
        axis_color = "silver"
        ax.spines['bottom'].set_color(axis_color)
        ax.spines['top'].set_color(axis_color) 
        ax.spines['right'].set_color(axis_color)
        ax.spines['left'].set_color(axis_color)
        ax.tick_params(axis='x', colors=axis_color)
        ax.tick_params(axis='y', colors=axis_color)

        i += 1

i_bottom_left_graph = len(graph_type_order) * len(my_range) 
ax_bottom_left = axes_flatten[i_bottom_left_graph]
ax_bottom_left.set_ylabel("articles\n(millions)");
ax_bottom_left.set_xlabel("year of publication");
axis_color = "black"
ax_bottom_left.spines['bottom'].set_color(axis_color)
ax_bottom_left.spines['top'].set_color(axis_color) 
ax_bottom_left.spines['right'].set_color(axis_color)
ax_bottom_left.spines['left'].set_color(axis_color)
ax_bottom_left.tick_params(axis='x', colors=axis_color)
ax_bottom_left.tick_params(axis='y', colors=axis_color)


# **{{print figure_link("small-multiples-num-papers-past")}}: Articles by year of observation, 2014-2018.**  Each row is an OA Type, each column is a Year of Observation, the x-axis of each graph is the Year of Publication, and the y-axis is the total number of articles (millions) available at the year of observation.

# In[35]:


first_year_row = 2014


# We can see that Gold, Hybrid, and Immediate Bronze OA articles all simply accumulate new articles each year, immediately.  For example, the {{ print first_year_row+1}} Gold graph is identical to the  {{ print first_year_row}} Gold graph beside it, other than the addition of a new, taller rightmost bar showing new papers published and made available in 2015.  
# 
# In contrast, Green OA (6th row) and Delayed Bronze OA (2nd row) graphs all have more complicated trends.  The graphs for the {{ print first_year_row+1}} observation year differ from the {{ print first_year_row}} graphs beside them in that they have a few new publications in {{ print first_year_row+1}}, but they also boost the {{ print first_year_row}} publication year, and even older years.  In fact we can see that when observed in {{ print first_year_row+4}} (the last column of the whole figure) Green OA is higher in all publication years than it was in the observation year {{ print first_year_row}} (the first column in the figure) because of met embargoes and backfilling.  A similar trend is visible for Delayed Bronze OA.  

# It is hard to see at the scale of {{print figure_link("small-multiples-num-papers-past")}}, but the Closed access graphs (top row) have the opposite trend -- when observed in 2018 (the last column), *fewer* papers in early bars of the histogram were considered Closed OA compared to an observation made in {{ print first_year_row}} (first column).  This is because some of what was "observed" as Closed in {{ print first_year_row}} has become Green and Bronze by the observation year of {{ print first_year_row+4}}, and therefore no longer appears in the Closed access histograms.

# <a id="section-4-1-6"></a>
# #### 4.1.6 Combined Past OA by date of observation

# In[36]:


register_new_figure('articles_by_oa_historical');


# We can now graph all papers by OA availability, by taking the area under each histogram in {{print figure_link("small-multiples-num-papers-past")}}.  This gives us {{print figure_link("articles_by_oa_historical")}}, with the absolute number of articles on the left panel and proportion by OA type on the right panel.  We can see that the number of OA articles has been growing over time between 2000 and 2018, though slowly.  Looking at the last year of observation in the proportion graph and its table below shows 27% of the literature published by 2018 could be observed as OA in 2018.

# In[37]:


articles_by_obs_year_df = papers_per_year_historical.copy()
articles_by_obs_year_df = articles_by_obs_year_df.rename(
    columns={"prediction_year": "x", "num_articles": "y"})
(articles_by_obs_historical, articles_by_obs_historical_proportional) = plot_area_and_proportion(
    articles_by_obs_year_df, 
     "standard", 
     2000, 2018, 2018,
     xlabel="year of observation", 
    fancy="cumulative");


# **{{print figure_link("articles_by_observation_year_prediction")}}: Total articles by OA type, by year of observation.** OA type as of year of observation.

# Table of percentages for the right panel:

# In[102]:


df = articles_by_obs_historical_proportional.copy()
rows = df.loc[(df.index==2010) | (df.index==2018)]
rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# <a id="section-4-2"></a>
# ### 4.2 Future OA Publication, by date of observation
# 
# <a id="section-4-2-1"></a>
# #### 4.2.1 Approach

# We wish to project OA availability {{print figure_link("small-multiples-num-papers-past")}} in future years.  How can we extrapolate these graphs into the future?
# 
# The model we use is based on observing that the papers that become available each year have a consistent distribution by article age, as seen in {{print figure_link("oa_lag_green")}} for Green OA and {{print figure_link("oa_lag_delayed_bronze")}} for Bronze OA (the histograms within each figure have a similar shape).  
# 
# If we then assume that the articles that will become available next year are similar to the articles that became available this year, for a given article age and OA type we can predict the future like this:
# 
# ```
#     total articles available next year  = 
# 
#         total articles available this year
# 
#          +
# 
#          scaling factor to account for growth
#          *
#          articles made newly available last year```

# In[39]:


register_new_figure('extrap_linear');
register_new_figure('extrap_exp');
register_new_figure('extrap_negative_exp');


# We have much of what we need already calculated in previous sections:  the **total articles available this year** is the observation year 2018 in {{print figure_link("small-multiples-num-papers-past")}}, and the **articles made newly available last year** is the last histogram of  {{print figure_link("oa_lag_green")}} for Green OA and {{print figure_link("oa_lag_delayed_bronze")}} for Bronze OA.  
# 
# All that remains is to calculate the **scaling factor to account for growth**.  We do this next.

# <a id="section-4-2-2"></a>
# #### 4.2.2 Scaling factor
# 
# {{print figure_link("extrap_linear")}} shows a scatter plot of new articles by OA type, by year of observation.  We add a linear best fit line, using the `scipy.optimize.curve_fit()` function.  The r<sup>2</sup> value below each graph is the sum of squares between the data and the fit, indicating goodness of fit (close to 1.0 is better).    

# In[40]:


naive_data_all = pd.DataFrame()

curve_type="linear"
fig, axes = plt.subplots(1, len(graph_type_order), figsize=(12, 2), sharex=True, sharey=False)
axes_flatten = axes.flatten()
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

for i, graph_type in enumerate(graph_type_order):
    curve_type_display = curve_type
    data_for_plot = papers_per_year_historical.loc[papers_per_year_historical.graph_type==graph_type]
    new_data = curve_fit_with_ci(graph_type, data_for_plot, curve_type=curve_type_display, ax=axes_flatten[i])
    new_data["curve_type"] = curve_type
    new_data["graph_type"] = graph_type
    naive_data_all = naive_data_all.append(new_data)

plt.show()


# **{{print figure_link("extrap_linear")}}: Total articles by year of observation, by OA type, with a linear extrapolation.**  

# We can see this isn't a particularly good fit for any of the OA types, so we try fitting with an exponential curve e<sup>x</sup> in {{print figure_link("extrap_exp")}}.  This is a better fit for the first four OA types, which we can see both visually and because they have higher r<sup>2</sup> values.

# In[41]:


curve_type="exp"
fig, axes = plt.subplots(1, len(graph_type_order), figsize=(12, 2), sharex=True, sharey=False)
axes_flatten = axes.flatten()
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

for i, graph_type in enumerate(graph_type_order):
    curve_type_display = curve_type
    data_for_plot = papers_per_year_historical.loc[papers_per_year_historical.graph_type==graph_type]
    new_data = curve_fit_with_ci(graph_type, data_for_plot, curve_type=curve_type_display, ax=axes_flatten[i])
    new_data["curve_type"] = curve_type
    new_data["graph_type"] = graph_type
    naive_data_all = naive_data_all.append(new_data)

plt.show()


# **{{print figure_link("extrap_exp")}}: Total articles by year of observation, by OA type, with an exponential extrapolation.**  

# The Delayed Bronze and Closed data look like they may actually trend down, so something of the form 1 - e<sup>x</sup> may be a better fit.  This is shown in  {{print figure_link("extrap_negative_exp")}} for all OA Types and does indeed look like the best fit yet for both Delayed Bronze and Closed.

# In[42]:


curve_type="negative_exp"
fig, axes = plt.subplots(1, len(graph_type_order), figsize=(12, 2), sharex=True, sharey=False)
axes_flatten = axes.flatten()
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

for i, graph_type in enumerate(graph_type_order):
    curve_type_display = curve_type
    data_for_plot = papers_per_year_historical.loc[papers_per_year_historical.graph_type==graph_type]
    new_data = curve_fit_with_ci(graph_type, data_for_plot, curve_type=curve_type_display, ax=axes_flatten[i])
    new_data["curve_type"] = curve_type
    new_data["graph_type"] = graph_type
    naive_data_all = naive_data_all.append(new_data)

plt.show()


# **{{print figure_link("extrap_negative_exp")}}: Total articles by year of observation, by OA type, with an exponential extrapolation fitting 1-exp().**  

# We conclude this hunt for the best scaling factors by choosing the extrapolation function with the highest r<sup>2</sup> value for each OA Type.  We use the chosen curves to extrapolate through 2025, and use the ratio of the value in 2018 to each subsequent observation year as that year's scaling factor.

# In[43]:


final_extraps = pd.DataFrame()
final_extraps = final_extraps.append(naive_data_all.loc[(naive_data_all.graph_type == "delayed_bronze") & (naive_data_all.curve_type=="negative_exp")])
final_extraps = final_extraps.append(naive_data_all.loc[(naive_data_all.graph_type == "closed") & (naive_data_all.curve_type=="negative_exp")])
final_extraps = final_extraps.append(naive_data_all.loc[(naive_data_all.graph_type != "delayed_bronze") &
                                                        (naive_data_all.graph_type != "closed") & 
                                                        (naive_data_all.curve_type=="exp")])


# In[44]:


register_new_figure('small-multiples-num-papers-future');


# <a id="section-4-2-3"></a>
# #### 4.2.3 Future OA by date of observation, by date of publication
# 
# We now have all the information we need to calculate **total articles available next observation year** as described in [Section 4.2.1](#section-4-2-1).  We use this approach to calculate total articles available at observation year 2019 based on 2018 data, then apply it again to calculate total articles at observation year 2020, and so on until 2025.  The result is shown in {{print figure_link("small-multiples-num-papers-future")}}.

# In[45]:


def get_all_predicted_papers(my_min, my_max):
    all_predicted_papers = pd.DataFrame()
    for i, graph_type in enumerate(graph_type_order):
        all_data = get_papers_by_availability_year_including_future(graph_type, my_min, my_max)
        all_data["graph_type"] = graph_type
        all_predicted_papers = all_predicted_papers.append(all_data)
    return all_predicted_papers

get_ipython().magic(u'cache all_predicted_papers_future = get_all_predicted_papers(1995, 2026)')


# In[46]:


my_range = range(2020, 2025+1)
fig, axes = plt.subplots(len(graph_type_order)+1, len(my_range), figsize=(12, 6), sharex=True, sharey=False)
axes_flatten = axes.flatten()
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

i = 0
for observation_year in my_range:
    ax = axes_flatten[i]
    ax.set_axis_off() 
    column_label = "observation year\n{}".format(observation_year)
    ax.text(.3, .2, column_label,
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=14,
        transform=ax.transAxes)
    i += 1

for graph_type in graph_type_order[::-1]:
    for observation_year in my_range:    
        ax = axes_flatten[i]
        this_data = all_predicted_papers_future.copy()
        this_data = this_data.loc[this_data.graph_type == graph_type]
        this_data = this_data.loc[this_data.prediction_year == observation_year]
        this_data["publication_date"] = [int(observation_year - a) for a in this_data.article_years_from_availability]
        new_data = graph_available_papers_in_observation_year_by_pubdate(graph_type, this_data, observation_year, ax=ax)

        y_max = all_predicted_papers_future.loc[(all_predicted_papers_future.graph_type == graph_type) &
                                               (all_predicted_papers_future.prediction_year <= max(my_range))]["num_articles"].max()
        ax.set_ylim(0, 1.2*y_max)
        
        axis_color = "silver"
        ax.spines['bottom'].set_color(axis_color)
        ax.spines['top'].set_color(axis_color) 
        ax.spines['right'].set_color(axis_color)
        ax.spines['left'].set_color(axis_color)
        ax.tick_params(axis='x', colors=axis_color)
        ax.tick_params(axis='y', colors=axis_color)

        i += 1

i_bottom_left_graph = len(graph_type_order) * len(my_range)
ax_bottom_left = axes_flatten[i_bottom_left_graph]
ax_bottom_left.set_ylabel("articles\n(millions)");
ax_bottom_left.set_xlabel("year of publication");
axis_color = "black"
ax_bottom_left.spines['bottom'].set_color(axis_color)
ax_bottom_left.spines['top'].set_color(axis_color) 
ax_bottom_left.spines['right'].set_color(axis_color)
ax_bottom_left.spines['left'].set_color(axis_color)
ax_bottom_left.tick_params(axis='x', colors=axis_color)
ax_bottom_left.tick_params(axis='y', colors=axis_color)


# In[ ]:





# **{{print figure_link("small-multiples-num-papers-future")}}: Articles by year of observation, extrapolated into the future.**   Each row is an OA Type, each column is a Year of Observation, the x-axis of each graph is the Year of Publication, and the y-axis is the total number of articles (millions) available at the year of observation.

# <a id="section-4-2-4"></a>
# #### 4.2.4 Combined Future OA by date of observation

# Finally, as in [Section 4.1.6](#section-4-1-6), we can sum the area under the histograms above to calculate total articles by year of observation by OA Type, for past articles as well as future projections.  We show this in {{print figure_link("articles_by_observation_year_prediction")}}.

# In[47]:


register_new_figure("articles_by_observation_year_prediction");
articles_by_obs_year_df = all_predicted_papers_future.copy()
articles_by_obs_year_df = articles_by_obs_year_df.rename(
    columns={"prediction_year": "x", "num_articles": "y"})
(df_articles_absolute, df_articles_proportional) = plot_area_and_proportion(articles_by_obs_year_df, 
                         "standard", 
                         2000, 2025, 2018,
                         xlabel="year of observation")


# **{{print figure_link("articles_by_observation_year_prediction")}}: Total articles by OA type, by year of observation.** OA type as of year of observation.

# We project 44% of articles will be OA by 2025: Gold will account for 15% of all articles, Bronze 13%, and Green and Hybrid 7% each.  A table showing the proportions of the right panel is below:

# In[48]:


df = df_articles_proportional.copy()
rows = df.loc[(df.index==2010) | (df.index==2019) | (df.index==2025)]
rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# 
# If we plot the difference between observation years in {{print figure_link("articles_by_observation_year_prediction")}}, we get the *net change* in articles by OA type, by year of observation.  This net change is shown in {{print figure_link("articles_by_observation_year_prediction_diff")}}.   

# In[49]:


register_new_figure("articles_by_observation_year_prediction_diff");
articles_by_obs_year_df = all_predicted_papers_future.copy()
articles_by_obs_year_df = articles_by_obs_year_df.rename(
    columns={"prediction_year": "x", "num_articles": "y"})

# articles_by_obs_year_df_closed = articles_by_obs_year_df.loc[
#     (articles_by_obs_year_df.graph_type=="closed") & 
#     (articles_by_obs_year_df.x <= 2025)]
# print articles_by_obs_year_df_closed
# plt.plot(articles_by_obs_year_df_closed.groupby("x").y.sum())
# plt.ylim(0, 2000000)

# plt.plot(articles_by_obs_year_df_closed.groupby("x").y.sum().diff())
# plt.ylim(0, 2000000)
num_articles_diff, num_articles_diff_proportional = plot_area_and_proportion(articles_by_obs_year_df, 
                         "standard", 
                         2000, 2025, 2018,
                         xlabel="year of observation", 
                         fancy="diff")


# **{{print figure_link("articles_by_observation_year_prediction_diff")}}: Change in number of articles from previous year of observation, by OA type.** Includes newly published articles, as well as articles that have changed OA type.

# This shows that by 2025 72% of articles that are newly available every year are available as OA, compared to 52% in 2019.  About half of the articles that become OA each year are Gold.

# In[50]:


df = num_articles_diff_proportional.copy()
rows = df.loc[(df.index==2010) | (df.index==2019) | (df.index==2025)]
rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# <a id="section-4-3"></a>
# ### 4.3 Past and Future OA Views

# <a id="section-4-3-1"></a>
# #### 4.3.1 Approach

# Now that we have projections of publication trends, we change tack to examine *views* -- when people access the literature, how likely is it the article they want to read is available as OA?  How do we think this has changed over the years, and what patterns do we project in the future?
# 
# To answer these questions we will use data from the Unpaywall browser extension, as described in [Section 2.2](#section-2-2) above. This data allows us to make inferences about overall readership trends.

# We will estimate views using this general equation:
# 
# ***
# ```
#  views =  (number of articles) * (views/article)
# ```
# ***
# 
# A key assumption underlying our model is that the **views/article by age of article** distribution curve is stable over time, for each OA type.  We calculate this distribution for views made during July 2018, and assume that readers in all other months and years, past and future, will have a similar relative interest in articles based on their age and OA type -- we assume the number of views varies solely based on number of articles available of each age and OA type.
# 
# Because we want to know views over time (rather than just at a single point in time) we treat each of the terms in the above equation as a [signal](https://en.wikipedia.org/wiki/Digital_signal) and use [digital signal processing](https://en.wikipedia.org/wiki/Digital_signal_processing) calculation techniques, which we will describe as we encounter them and in supplementary information.
# 
# The signals for **number of articles** were already calculated in Sections 4.1 and 4.2.
# 
# The signals for **views/article** will be calculated in Section 4.3.2.  
# 
# We "multiply" these two signals together using signal processing techniques, described in Section 4.3.3, to get total views across time.
# 
# We do these calculations for each OA type individually, and then add them together in Section 4.3.5 to look at relative trends.
# 
# 

# <a id="section-4-3-2"></a>
# #### 4.3.2 Views per article

# We calculate views per article as you'd expect:
# ```
#     the average number of views per article  = 
# 
#         (the total number of views for articles of that age and OA type) 
# 
#          /
# 
#         (the number of articles of that age and OA type)
# ```

# We can state this more concisely and precisely as follows.  For each OA type:
# 
# ```
# views per article by age  = dot_division( views by age, articles by age )
#     
# ```
# 
# where dot_division is the [element-wise Hadamard division](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) of two signals.

# In[51]:


register_new_figure("view-by-age-no-color");


# It is important to look at views by article age, because it is well known that readers are more interested in accessing newly-published articles, and indeed this trend can be seen in the Unpaywall usage logs. {{ print figure_link("view-by-age-no-color") }}  shows monthly access requests to the Unpaywall extension made between August 2018 and August 2019, distributed by article age. As expected, the distribution is very skewed and readers are most interested in articles published less than a year ago.
# 

# In[52]:


# hidden: code to query and graph 
get_ipython().magic(u'matplotlib inline')

my_data = views_by_age_months_no_color_full_year.loc[views_by_age_months_no_color_full_year.article_age_months >= 0]
my_data = my_data.loc[my_data.article_age_months <= 12*15]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2), sharex=True, sharey=False)
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1, wspace=.3)

my_plot = my_data.plot.line(x="article_age_months", y="num_views", ax=ax1)
ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.1f}'.format(y/(1000*1000.0))))
ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '{0:,.1f}'.format(x/(12.0))))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('article age (years)')
ax1.set_ylabel('views (millions)')
ax1.get_legend().remove()

my_plot = my_data.plot.line(x="article_age_months", y="num_views", ax=ax2)
ax2.set_yscale("log")
ax2.set_xlabel('article age (years)')
ax2.set_ylabel('views (log scale)');
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_legend().remove()


# **{{print figure_link("view-by-age-no-color")}}: Distribution of views by article age**: left panel is linear y-axis and y panel is a log plot.

# In[53]:


register_new_figure("views_by_age_with_color");


# We now look at this data by OA type.  To simplify interactions, for the rest of the analysis we will restrict our view data to that from just one month: July 2019, and to articles less than 15 years old.  This accounts for {{ print "{0:,.0f}".format(views_per_year_total.num_views_one_month.sum()) }} views, which we then multiply by 12 to approximate a year's worth of views by the Unpaywall extension (not important because the point of the views analysis is growth rather than absolute numbers, but it helps slightly with interpretation).  
# 
# {{ print figure_link("views_by_age_with_color") }} shows the distribution of views by article age, by OA type.

# In[54]:


get_ipython().magic(u'cache views_per_year_total = get_views_per_year_total()')
data_now = views_per_year_total.loc[views_per_year_total["article_age_years"] >= 0]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(linewidth=5)
g.map(plt.plot, "article_age_years", "num_views_per_year", **kws);
for ax in g.axes.flat: 
    ax.set_xlabel("article age (years)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.1f}'.format(y/(1000*1000.0))))
g.axes.flat[0].set_ylabel("views per year (millions)");
    


# **{{print figure_link("views_by_age_with_color")}}: Distribution of views by article age, by OA type**

# Closed access articles receive the most views, which isn't surprising because as we saw in [Section 4.1.6](#section-4-1-6), most articles available in 2018 are Closed access.  What happens when we divide these curves to get views *per article*, as is our goal?  
# 
# A detailed walkthrough is given in Supplementary Information [Section 11.4](#section-11-4).  Here we present the results of dividing the above view distribution signals by the article signals we calculated {{print figure_link("small-multiples-num-papers-future")}} for the 2018 observation year:

# In[55]:


register_new_figure("views-by-article-main");


# In[56]:


def get_views_per_article(graph_type):
    if graph_type == "biorxiv":
        graph_type = "green"
        
    views_per_year = get_views_per_year(graph_type)
    papers_per_year = get_papers_by_availability_year(graph_type, 2018, just_this_year=False)
    papers_per_year["article_age_years"] = papers_per_year["article_years_from_availability"]
    papers_per_year = papers_per_year.loc[(papers_per_year["article_age_years"] <=15 )]

    data_merged_clean = papers_per_year.merge(views_per_year, on=["article_age_years"])        
    data_merged_clean["views_per_article"] = data_merged_clean["num_views_per_year"] / data_merged_clean["num_articles"]

    views_per_article = pd.DataFrame(data_merged_clean, columns=["article_age_years", "views_per_article"])
    views_per_article = views_per_article.sort_values(by="article_age_years")

    if graph_type=="delayed_bronze":
        # otherwise first one is too high because number articles too low in year 0 for delayed subset
        views_per_article.loc[views_per_article.article_age_years==0, ["views_per_article"]] = float(views_per_article.loc[views_per_article.article_age_years==1].views_per_article)

    return views_per_article

def get_views_per_article_total():
    all_data = pd.DataFrame()
    for prep_graph_type in ["gold", "hybrid", "green", "immediate_bronze", "delayed_bronze", "closed"]:
        temp_papers = get_views_per_article(prep_graph_type)
#         print prep_graph_type
#         print "{:,.0f}".format(temp_papers.views_per_article.max()), "{:,.0f}".format(temp_papers.views_per_article.sum())
#         print "\n"
        temp_papers["graph_type"] = prep_graph_type
        all_data = all_data.append(temp_papers)
    return all_data


# In[57]:



get_ipython().magic(u'cache views_per_article_total = get_views_per_article_total()')
data_now = views_per_article_total.loc[views_per_article_total["article_age_years"] >= 0]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(s=50)
g.map(plt.scatter, "article_age_years", "views_per_article", **kws);
g.map(plt.plot, "article_age_years", "views_per_article");
for ax in g.axes.flat: 
    ax.set_xlabel("article age (years)")
g.axes.flat[0].set_ylabel("views per article");


# **{{print figure_link("views-by-article-main")}}: Views by article curve, by OA type**

# We see in {{print figure_link("views-by-article-main")}} that number of views per article is much higher for Green than other kinds of articles, particularly for articles that are available as Green OA within the first year of publication (age 0).  This is consistent with previously-documented download advantages of Green OA articles, and could be caused by various factors including self-selection bias, or the common cause of strong funding support for high-interest medical papers by funders like the NIH.
# 
# Relative to Closed access articles, the average number of views per article for Gold, Hybrid, and Delayed Bronze is particularly strong for older articles.

# <a id="section-4-3-3"></a>
# #### 4.3.3 Calculating Views

# Finally, we are ready to calculate overall views. As a reminder, here is our general approach:  
# ***
# ```
#  views =  (number of articles) * (views/article)
# ```
# ***
# 
# We can state this more precisely as follows.  For each OA type:
# 
# ```
#  views in a given year = convolution(articles by age for that year, 
#                                      views/article by age)     
# ```
# 
# where [convolution](https://en.wikipedia.org/wiki/Convolution) is the standard mathematical operation of modifying a signal by another signal, by integrating the product of the two curves after one is reversed and shifted.  

# A detailed walkthrough of this convolution (and more information on what convolution means!) is given in Supplementary Information [Section 11.5](#section-11-5).  Here we present the results of convolution, which can be seen roughly as multiplication of the articles-by-age estimates we made in [Section 4.2.2](#section-4-2-2) and  [Section 4.2.3](#section-4-2-3) with the views/article curves we calculated in [Section 4.3.2](#section-4-3-2).  

# In[58]:


def get_predicted_views(graph_type, now_delta_years=0, label_for_graph=None, show_graph=True):
#     calc_min_year = 1951
    calc_min_year = 1995
    display_min_year = 2010
    now_year = 2020 - now_delta_years
    max_year = 2025
    exponential = False

    if graph_type == "biorxiv":
        exponential = True
        
    views_per_article = get_views_per_article(graph_type)
           
    df_views_by_year = pd.DataFrame()
    all_papers_per_year = get_papers_by_availability_year_including_future(graph_type, calc_min_year, max_year)
    for prediction_year in range(calc_min_year, max_year+1):        
#     for prediction_year in range(calc_min_year, 2019):        
#     for prediction_year in range(2017, 2019):        
        papers_per_year = all_papers_per_year.loc[all_papers_per_year["prediction_year"] == prediction_year]
#         print views_per_article.head()
        try:
            data_merged_clean = papers_per_year.merge(views_per_article, left_on=["article_years_from_availability"], right_on=["article_age_years"])
            data_merged_clean = data_merged_clean.sort_values("article_age_years")
            win = data_merged_clean["views_per_article"] 
            sig = data_merged_clean["num_articles"]
            views_by_observation_year = signal.convolve(win, sig, mode='same', method="direct")
            y = max(views_by_observation_year)
            df_views_by_year = df_views_by_year.append(pd.DataFrame({"observation_year":[prediction_year], "views": [y]}))
        except (ValueError, KeyError):  # happens when the year is blank
            pass
        

    return df_views_by_year

def get_predicted_views_total(observation_year):
    all_data = pd.DataFrame()
    for prep_graph_type in graph_type_order:
        temp_papers = get_predicted_views(prep_graph_type, observation_year)
        temp_papers["graph_type"] = prep_graph_type
        all_data = all_data.append(temp_papers)
    return all_data


# In[59]:


register_new_figure("views-small-main");


# In[60]:


observation_year = 2025
get_ipython().magic(u'cache predicted_views_total = get_predicted_views_total(observation_year)')

data_now = predicted_views_total.loc[predicted_views_total["observation_year"] >= 2010]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(marker="x", s=70)
g.map(plt.scatter, "observation_year", "views", **kws);
for ax in g.axes.flat: 
    ax.set_xlabel("observation year")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.0f}'.format(y/(1000*1000.0))))
g.axes.flat[0].set_ylabel("views (millions)");


# **{{print figure_link("views-small")}}: Views** Views by year.

# <a id="section-4-3-4"></a>
# #### 4.3.4 Combined Past and Future Views

# We can plot these lines stacked on top of each other to see how the OA types change over time, shown in {{print figure_link("views_stacked")}}.
# 

# In[61]:


register_new_figure("views_stacked");


# In[62]:



# not cumulative because cumulative views don't mean anything

views_all_data_pivot = predicted_views_total.pivot_table(
             index='observation_year', columns='graph_type', values=['views'], aggfunc=np.sum)\
       .sort_index(axis=1, level=1)\
       .swaplevel(0, 1, axis=1)
views_all_data_pivot.columns = views_all_data_pivot.columns.levels[0]
views_all_data_pivot
# all_data_pivot[oa_status_order].plot.area(stacked=True, color=oa_status_colors)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=False)
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

views_all_data_pivot_graph = views_all_data_pivot.copy()
views_all_data_pivot_graph = views_all_data_pivot_graph.loc[views_all_data_pivot_graph.index > 1960]
my_plot = views_all_data_pivot_graph[graph_type_order].plot.area(stacked=True,  linewidth=.1, color=graph_type_colors, ax=ax1)
ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.0f}'.format(y/(1000*1000.0))))
ax1.set_xlabel('year of view')
ax1.set_ylabel('views (millions)')
ax1.set_xlim(2000, 2025)
ax1.set_ylim(0, 1.2*max(views_all_data_pivot_graph.sum(1)))
ax1.set_title("Estimated views by year of observation");
handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:6]), reversed(labels[0:6]), loc='upper left');  # reverse to keep order consistent

views_df_diff_proportional = views_all_data_pivot_graph.div(views_all_data_pivot_graph.sum(1), axis=0)
my_plot = views_df_diff_proportional[graph_type_order].plot.area(stacked=True,  linewidth=.1, color=graph_type_colors, ax=ax2)
my_plot.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax2.set_xlabel('year of view')
ax2.set_ylabel('proportion of views')
ax2.set_title("Proportion of views");
ax2.set_xlim(2000, 2025)
ax2.set_ylim(0, 1)
handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:6]), reversed(labels[0:6]), loc='upper left');  # reverse to keep order consistent

plt.tight_layout(pad=.5, w_pad=4, h_pad=2.0)  




# **{{print figure_link("views_stacked")}}: Views, by year of view**

# Here is the raw data for the proportions at a few specific observation years:

# In[103]:


df = views_df_diff_proportional.copy()
rows = df.loc[(df.index==2000) | (df.index==2010) | (df.index==2019) | (df.index==2025)]
rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# Our estimated number of views per year increases steadily over time, and the proportion of views to OA resources goes from 16% in 2000 to 52% in 2019, and to 70% in 2025.  This increase is driven primarily by Green and Gold OA: we estimate 19% of views in 2025 will lead to Green OA articles and 33% of views will lead to Gold articles.  

# In[133]:


# HAP get total absolute view coefficients

# df = views_all_data_pivot_graph.copy()
# rows = df.loc[(df.index>=2018)]
# rows["total views"] = rows.sum(1)
# rows["total views percent of 2018"] = rows["total views"] / float(rows.loc[rows.index==2018]["total views"])
# rows["total oa views"] = rows["total views"] - rows["closed"]
# rows["total oa views percent of 2018"] = rows["total oa views"] / float(rows.loc[rows.index==2018]["total oa views"])

# # print rows.head()
# # print rows
# my_markdown = tabulate(rows[["total views", "total views percent of 2018", "total oa views", "total oa views percent of 2018"]], tablefmt="pipe", headers="keys", floatfmt=",.2f")
# display(Markdown(my_markdown))


# <a id="section-4-4"></a>
# ### 4.4 Extending the model:  Growth of bioRxiv

# An advantage of building a model is that now we can layer on alternate assumptions and see how anticipated disruptions might affect OA in coming years. A comprehensive examination of all the alternative futures is clearly outside the scope of this paper, however an example will be illustrative.
# 
# BioRxiv, a preprint server in biology, provides an excellent example.  As described in Abdill and Blekhman (2019), deposits into bioRxiv are growing rapidly. If growth continues at the current rate, biorxiv could prove to be a major disruptor: it is growing extremely quickly, and the vast majority of the deposits which are published have zero OA lag (and so are OA at the time of highest demand).

# We model the growth of bioRxiv and its impact on OA availability by extrapolating from bioRxiv papers that:
# 
# -   were deposited at or before the date they were published (to simplify the model), and
# -   are Closed access other than their Green bioRxiv copy (so that we don't double-count articles made OA as Gold, Hybrid, or Bronze).
# 
# The number of articles that meet these criteria are shown in the table below, by date of publication.

# In[104]:


biorxiv_growth_otherwise_closed


# In[65]:


register_new_figure("biorxiv-exp");


# This growth has a very strong logarithmic extrapolation fit, as seen in {{print figure_link("biorxiv-exp")}}.  

# In[66]:


biorxiv_now_year = 2018

# reset
papers_per_year_historical = papers_per_year_historical.loc[papers_per_year_historical.graph_type != 'biorxiv']

for graph_type in ["biorxiv"]:
    for prediction_year in range(2000, biorxiv_now_year+1):        
        papers_per_year = get_papers_by_availability_year(graph_type, prediction_year, just_this_year=True)
        papers_per_year["graph_type"] = graph_type
        papers_per_year["prediction_year"] = prediction_year
        papers_per_year_historical = papers_per_year_historical.append(papers_per_year)


# In[67]:


fig, ax = plt.subplots(1, 1, figsize=(4, 2), sharex=True, sharey=False)
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)
    
data_for_plot = papers_per_year_historical.loc[papers_per_year_historical.graph_type=="biorxiv"]
new_data = curve_fit_with_ci("biorxiv", data_for_plot, curve_type="exp", ax=ax)
new_data["curve_type"] = "exp"
new_data["graph_type"] = "biorxiv"
final_extraps = final_extraps.loc[final_extraps.graph_type != 'biorxiv']
final_extraps = final_extraps.append(new_data)
ax.set_xlim(2012, 2025)
ax.set_yscale("log")
ax.set_ylabel("articles (log scale)");


# **{{print figure_link("biorxiv-exp")}}: bioRxiv extrapolation**

# In[68]:


register_new_figure("articles_by_observation_year_prediction_plus_biorxiv");


# BioRxiv won't be able to grow exponentially forever -- there are a limited number of papers in Biology.  But if we were to imagine bioRxiv continued its current growth rate for another 5 years, we would estimate its impact on the  relative proportion of articles available as OA in {{ print figure_link("articles_by_observation_year_prediction_plus_biorxiv")}}.

# In[69]:


all_predicted_papers_future_plus_biorxiv = all_predicted_papers_future.copy()

biorxiv_predicted_papers = get_papers_by_availability_year_including_future("biorxiv", 1995, 2026)
biorxiv_predicted_papers["graph_type"] = "biorxiv"

all_predicted_papers_future_plus_biorxiv = all_predicted_papers_future_plus_biorxiv.append(biorxiv_predicted_papers)


articles_by_obs_year_df_plus_biorxiv = all_predicted_papers_future_plus_biorxiv.copy()
articles_by_obs_year_df_plus_biorxiv = articles_by_obs_year_df_plus_biorxiv.rename(
    columns={"prediction_year": "x", "num_articles": "y"})
plot_area_and_proportion(articles_by_obs_year_df_plus_biorxiv, 
                         "standard_plus_biorxiv", 
                         2000, 2025, 2018,
                         xlabel="year of observation");


# **{{print figure_link("articles_by_observation_year_prediction_plus_biorxiv")}}: Prediction of articles by OA type by year of observation, including bioRxiv**

# In[70]:


register_new_figure("biorxiv-stacked");


# This doesn't look like many articles, but let's see how it affects viewership.  For simplicity we use the generic green OA access trend derived in {{print figure_link("views-by-article") }}.  This results in views as shown in {{print figure_link("biorxiv-stacked") }} -- a notable impact on the total views of all scholarly papers, and obviously the impact would be even greater within the field of biology.

# In[71]:


total_views_including_biorxiv = predicted_views_total.copy()
biorxiv_views = get_predicted_views("biorxiv", 2010, 2025)
biorxiv_views["graph_type"] = "biorxiv"
biorxiv_views["oa_status"] = "biorxiv"
total_views_including_biorxiv = total_views_including_biorxiv.append(biorxiv_views)


# In[72]:


all_data_pivot_plus_biorxiv = total_views_including_biorxiv.pivot_table(
             index='observation_year', columns='graph_type', values=['views'], aggfunc=np.sum)\
       .sort_index(axis=1, level=1)\
       .swaplevel(0, 1, axis=1)
all_data_pivot_plus_biorxiv.columns = all_data_pivot_plus_biorxiv.columns.levels[0]
# all_data_pivot_plus_biorxiv["biorxiv"] = all_data_pivot_plus_biorxiv["biorxiv"].fillna(0)
# all_data_pivot_plus_biorxiv["closed"] -= all_data_pivot_plus_biorxiv["biorxiv"]
all_data_pivot_plus_biorxiv

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=False)
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

all_data_pivot_plus_biorxiv_graph = all_data_pivot_plus_biorxiv
all_data_pivot_plus_biorxiv_graph = all_data_pivot_plus_biorxiv_graph.loc[all_data_pivot_plus_biorxiv_graph.index > 1960]
my_plot = all_data_pivot_plus_biorxiv_graph[graph_type_order_plus_biorxiv].plot.area(stacked=True, color=graph_type_colors_plus_biorxiv, ax=ax1, linewidth=0.1)
ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.0f}'.format(y/(1000*1000.0))))
ax1.set_xlabel('year of view')
ax1.set_ylabel('views (millions)')
ax1.set_xlim(2010, 2025)
ax1.set_ylim(0, 1.2*max(all_data_pivot_plus_biorxiv_graph.sum(1)))
ax1.set_title("Estimated views by year of observation, including biorxiv growth");
handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:7]), reversed(plus_biorxiv_labels[0:7]), loc='upper left');  # reverse to keep order consistent

df_diff_proportional_plus_biorxiv = all_data_pivot_plus_biorxiv.div(all_data_pivot_plus_biorxiv.sum(1), axis=0)
my_plot = df_diff_proportional_plus_biorxiv[graph_type_order_plus_biorxiv].plot.area(stacked=True, color=graph_type_colors_plus_biorxiv, ax=ax2, linewidth=0.1)
my_plot.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax2.set_xlabel('year of view')
ax2.set_ylabel('proportion of views')
ax2.set_title("Proportion of views, including biorxiv growth");
ax2.set_xlim(2010, 2025)
ax2.set_ylim(0, 1)
handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:7]), reversed(plus_biorxiv_labels[0:7]), loc='upper left');  # reverse to keep order consistent

plt.tight_layout(pad=.5, w_pad=4, h_pad=2.0) 
plt.subplots_adjust(hspace=1)


# **{{print figure_link("biorxiv-stacked")}}: Predicted views, including bioRxiv, by year of view**

# In[73]:


df = df_diff_proportional_plus_biorxiv.copy()
rows = df.loc[(df.index==2010) | (df.index==2019) | (df.index==2025)]
rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order_plus_biorxiv+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# 
# 
# ------------
# *Move this section to the top of the paper*
# 
# <a id="section-4-5"></a>
# ### Summary
# 

# Understanding the growth of open access (OA) is important for deciding funder policy, subscription allocation, and infrastructure planning.  
# 
# This study analyses the number of papers available as OA over time. The models includes both OA embargo data and the relative growth rates of different OA types over time, based on the OA status of 70 million journal articles published between 1950 and 2019.
# 
# The study also looks at article usage data, analyzing the proportion of views to OA articles vs views to articles which are closed access.  Signal processing techniques are used to model how these viewership patterns change over time.  Viewership data is based on 2.8 million uses of the Unpaywall browser extension in July 2019. 

# We found that Green, Gold, and Hybrid papers receive more views than their Closed or Bronze counterparts, particularly Green papers made available within a year of publication.  We also found that the proportion of Green, Gold, and Hybrid articles is growing most quickly.
# 
# In 2019: 
# - 31% of all journal articles are available as OA
# - 52% of article views are to OA articles
# 
# Given existing trends, we estimate that by 2025:
# - 44% of all journal articles will be available as OA
# - 70% of article views will be to OA articles
# 
# The declining relevance of closed access articles is likely to change the landscape of scholarly communication in the years to come.

# In[74]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=False)
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)

start_year = 2000
end_year = 2025
divide_year = 2018
xlabel="year of observation"
my_colors = graph_type_colors
my_color_order = graph_type_order
color_column = "graph_type"
fancy = None # "diff"


    
summary_views_pivot_actual = views_df_diff_proportional.loc[views_df_diff_proportional.index <= divide_year+1]
my_plot = summary_views_pivot_actual[graph_type_order].plot.area(stacked=True, color=my_colors, ax=ax1, linewidth=0.1)
summary_views_pivot_projected = views_df_diff_proportional.loc[views_df_diff_proportional.index > divide_year]
my_plot = summary_views_pivot_projected[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax1, alpha=0.6)
my_plot.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax1.set_xlabel('year of view')
ax1.set_ylabel('proportion of views')
ax1.set_xlim(2010, 2025)
ax1.set_ylim(0, 1)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(which='both', right='on', left='on')
ax1.yaxis.set_minor_locator(plt.MaxNLocator(10))
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_title("Projected views, by OA type")
handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:6]), reversed(labels[0:6]), loc='upper left');  # reverse to keep order consistent

summary_articles_pivot_actual = df_articles_proportional.loc[df_articles_proportional.index <= divide_year+1]
my_plot = summary_articles_pivot_actual[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax2)
summary_articles_pivot_projected = df_articles_proportional.loc[df_articles_proportional.index > divide_year]
my_plot = summary_articles_pivot_projected[my_color_order].plot.area(stacked=True, color=my_colors, linewidth=.1,  ax=ax2, alpha=0.6)
my_plot.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax2.set_xlabel(xlabel)
ax2.set_ylabel('proportion of articles')
#     ax2.set_title("Proportion of papers");
ax2.set_xlim(start_year, end_year)
ax2.set_ylim(0, 1)  
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(which='both', right='on', left='on')
ax2.yaxis.set_minor_locator(plt.MaxNLocator(10))
ax2.set_title("Projected articles, by OA type")
handles, labels = my_plot.get_legend_handles_labels(); my_plot.legend(reversed(handles[0:6]), reversed(labels[0:6]), loc='upper left');  # reverse to keep order consistent


plt.tight_layout(pad=.5, w_pad=4, h_pad=2.0) 
plt.subplots_adjust(hspace=1)


# Move the raw data below to supplementary information (it is the data behind the graphs above):

# Percent of views available as OA type, for certain years:

# In[105]:



df = views_df_diff_proportional
rows = df.loc[(df.index==2010) | (df.index==2019) | (df.index==2025)]
# with pd.option_context('display.float_format', '{:,.0f}%'.format):
#     print 100*rows[graph_type_order_plus_biorxiv]

rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# Percent of papers available as OA type, for certain years:

# In[106]:



df = df_articles_proportional.copy()
rows = df.loc[(df.index==2010) | (df.index==2019) | (df.index==2025)]
rows["all OA"] = 1 - rows["closed"]
my_markdown = tabulate(100*rows[graph_type_order+["all OA"]], tablefmt="pipe", headers="keys", floatfmt=",.0f")
display(Markdown(my_markdown))


# ---------------

# <a id="section-5"></a>
# ## 5. Discussion

# We found that Green, Gold, and Hybrid papers receive more views than their Closed or Bronze counterparts, particularly Green papers made available within a year of publication. We also found that the proportion of Green, Gold, and Hybrid articles is growing quickly.
# 
# In 2019: 
# 
# - 31% of all journal articles are available as OA
# - 52% of article views are to OA articles
# 
# Given existing trends, we estimate that by 2025:
# 
# - 44% of all journal articles will be available as OA
# - 70% of article views will be to OA articles

# Our model is conservative.  Although the extrapolations assume continued incremental adoption of OA, they do not yet model other disruptive changes that will likely increase the growth of OA in coming years: the adoption of Plan S, a change in embargo periods for existing mandates, a dramatic increase in institutional self-archiving, large scale read and publish agreements, etc.
# 
# This area is ripe for future research in other ways as well: understanding how OA publication and viewership rates vary by discipline, country, and publisher is key.  The assumption that the views/article curve is stable over time should be further investigated and relaxed if found to be inadequate.  The model could be refined in many other ways as well, for example to use custom viewership patterns for readers within a specific university, or of a specific journal.
# 
# One interesting realization from the modeling we've done is that when the proportion of papers that are OA increases, or when the OA lag decreases, the total number of views increase -- the scholarly literature becomes more heavily viewed and thus more valuable to society.  This is intuitive, but could be explored quantitatively in future work.

# The study has several limitations.  Only journal articles with DOIs are included, which under-represents disciplines and geographical areas which rely heavily on conference papers or articles without DOIs.  Illegal repositories (SciHub) or articles posted on academic social networks (ResearchGate and Academia.edu) are not considered, which may undercount articles that are relevant for some uses.  The users of the Unpaywall browser extension may not be representative of other readers, and using page views as a proxy for article interest is inexact. Nonetheless, we believe this analysis represents a useful approach for modeling the growth and importance of OA in the future.  

# The genesis for this study was a steady stream of inquiries from university librarians, asking for OA rates for specific journals to help inform their subscription decisions and negotiations.  We realized it would be even more helpful if we could provide OA rates (a) for the future, (b) by date when the OA resource is available, and (c) weighted by the importance of the article to their faculty.  The model presented here addresses these issues and will form the basis of information available to librarians and other decision-makers in the future.
# 
# The declining relevance of closed access articles is likely to change the landscape of scholarly communication in the years to come.

# ## 6. References

# - Abdill, R.J., and Blekhman, R. (2019). Tracking the popularity and outcomes of all bioRxiv preprints. eLife 8.
# 
# - Antelman, K. (2017). Leveraging the growth of open access in library collection decision making. At the Helm: Leading Transformation. Association of College and Research Libraries 411–422.
# 
# - Laakso, M., and Björk, B.-C. (2013). Delayed open access: An overlooked high-impact category of openly available scientific literature. Journal of the American Society for Information Science and Technology 64, 1323–1329.
# 
# - Lewis, D.W. (2012). The Inevitability of Open Access. College & Research Libraries 73, 493–506.
# 
# - Piwowar, H., Priem, J., Larivière, V., Alperin, J.P., Matthias, L., Norlander, B., Farley, A., West, J., and Haustein, S. (2018). The State of OA: a large-scale analysis of the prevalence and impact of Open Access articles. PeerJ 6, e4375.
# 
# - Piwowar, H., Priem, J., & Orr, R. (2019). Data From: The Future of OA: A large-scale analysis projecting Open Access publication and readership [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3474007

# <a id="section-7"></a>
# ## 7. Data and code availability
# 

# <a id="section-7-1"></a>
# ### 7.1 Empirical Gold OA list
# 
# The empirical Gold OA journal list is available in the Zenodo dataset at http://doi.org/10.5281/zenodo.3474007, in the file "gold_oa_empirical_list.csv".

# <a id="section-7-2"></a>
# ### 7.2 Empirical bronze delayed OA list
# 
# The empirical Bronze Delayed OA journal list is available in the Zenodo dataset at http://doi.org/10.5281/zenodo.3474007, in the file "delayed_bronze_empirical_list.csv".
# 
# The list of combined delayed OA policies we extracted from various sources is available in the  Zenodo dataset at http://doi.org/10.5281/zenodo.3474007, in the file "delayed_bronze_extracted_policies.csv".
# 

# <a id="section-7-3"></a>
# ### 7.3 Study data
# 
# All study data is available in Zenodo, at 
# ```
# Piwowar H, Priem J, & Orr R. (2019). Data From: The Future of OA: A large-scale analysis projecting Open Access publication and readership [Data set]. Zenodo.``` http://doi.org/10.5281/zenodo.3474007
# 

# <a id="section-7-4"></a>
# ### 7.4 Analysis notebook
# 
# The Jupyter analysis notebook is available from GitHub at https://github.com/Impactstory/future-oa.
# 
# Also, for Jupyter nerds and to help us remember: export using 
# ```jupyter nbconvert manuscript.ipynb --to html --TemplateExporter.exclude_input=True```
#  then push to github, then can be viewed at
# https://htmlpreview.github.io/?https://github.com/Impactstory/future-oa/blob/master/manuscript.html
# 

# ## 8. Competing Interests
# 
# The authors work at [Our Research](https://ourresearch.org/) (formerly Impactstory), a non-profit company that builds tools to make scholarly research more open, connected, and reusable, including Unpaywall.

# ## 9. Funding
# 

# The authors received no funding for this analysis.

# ## 10. Acknowledgements

# The authors would like to thank Bianca Kramer for extensive and valuable comments on a draft of this article.  The author order of JP and HP was determined by coin flip, as is their custom.

# <a id="section-11"></a>
# ## 11. Supplementary Information
# 

# <a id="section-11-1"></a>
# ### 11.1 Detailed look at OA Lag of Green OA
# 

# In[77]:


register_new_figure("detailed-green");


# This data supplements the discussion of Green OA lag in [Section 4.1.2](#section-4-1-2).

# In {{print figure_link("detailed-green")}} we plot the number of Green OA papers made available each year vs their date of publication. The first plot is a histogram of number of papers made available each year (one row for each year). The second plot is the same, but superimposes the articles made available in previous years. This stacked area represents the total cumulative number of Green OA papers that are available in that year -- if you were in that year and wondering what was available as Green OA that's what you'd find.
# 
# The third plot is a larger version of the availability as of 2018, showing the accumulation of availability. It allows us to appreciate that less than half of papers papers published in, say, 2015, were made available the same year -- most of the papers have been made available in subsequent years. The fourth plot is a slice in isolation, for clarity: the Green OA for articles with a Publication Date of 2015.

# In[78]:


make_detailed_plots("green")


# In[79]:


make_zoom_in_plot("green")


# **{{print figure_link("detailed-green")}}: Detailed Green OA**

# <a id="section-11-2"></a>
# ### 11.2 Detailed look at OA Lag of Delayed Bronze OA 

# In[80]:


register_new_figure("detailed-bronze");


# This data supplements the discussion of Delayed Bronze OA lag in [Section 4.1.3](#section-4-1-3).  
# 
# In {{print figure_link("detailed-bronze")}} we plot the number of Delayed Bronze OA papers made available each year vs their date of publication.  For more explanation see the text describing {{print figure_link("detailed-green")}} above.

# In[81]:


make_detailed_plots("delayed_bronze")


# In[82]:


make_zoom_in_plot("delayed_bronze")


# **{{print figure_link("detailed-bronze")}}: Detailed Delayed Bronze OA**

# <a id="section-11-3"></a>
# ### 11.3 Walk-through of dot division

# This is a walk-through of dot division, as discussed in [Section 4.3.2](#section-4-3-2).  For each OA type:
# 
# ```
# views per article by age  = dot_division( views by age, articles by age )
#     
# ```
# 
# where dot_division is the [element-wise Hadamard division](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) of two signals.

# For **views by age** we use {{print figure_link("views_by_age_with_color")}}, reproduced here:

# In[83]:


get_ipython().magic(u'cache views_per_year_total = get_views_per_year_total()')
data_now = views_per_year_total.loc[views_per_year_total["article_age_years"] >= 0]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(linewidth=5)
g.map(plt.plot, "article_age_years", "num_views_per_year", **kws);
for ax in g.axes.flat: 
    ax.set_xlabel("article age (years)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.1f}'.format(y/(1000*1000.0))))
g.axes.flat[0].set_ylabel("views per year (millions)");
    


# **{{print figure_link("views_by_age_with_color")}}: Views by OA type and article age**

# In[84]:


register_new_figure("num-papers-by-age-2018");


# For **articles by age** we use the curves we calculated in [Section 4.2.7](#section-4-2-7) above, specifically {{print figure_link("small-multiples-num-papers-future")}} for the 2018 observation year:

# In[85]:


get_ipython().magic(u'cache papers_by_availability_year_total_2018 = get_papers_by_availability_year_total(2018)')
data_now = papers_by_availability_year_total_2018.loc[papers_by_availability_year_total_2018["article_years_from_availability"] < 15]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
my_dict = {"width": 1, "edgecolor": (0,0,0,0)}
g.map(plt.bar, "article_years_from_availability", "num_articles", **my_dict);
for ax in g.axes.flat: 
    ax.set_xlabel("article age (years)")
    ax.set_xlim(0, 15)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.1f}'.format(y/(1000*1000.0))))
g.axes.flat[0].set_ylabel("articles (millions)");


# **{{print figure_link("num-papers-by-age-2018")}}: Articles by OA type and article age**

# Now for each of the OA types individually, we divide these signals by each other, element-wise. This means that, for each OA type, we divide the number of times someone viewed an article of that was 0 years old (in other words, published less than a year ago) by the number of total articles that were 0 years old -- articles that were published less than a year ago. Then we take the next age bucket, 1 years old, and divide the number of views of 1 year old articles by the number of articles available as that OA type that were 1 year old. We do this for all age bins (15 years are shown in the graphs).
# 
# The result of these divisions are the signals below: the number of views per article, for a given age and OA type.
# 

# In[86]:


register_new_figure("views-by-article");


# In[87]:



get_ipython().magic(u'cache views_per_article_total = get_views_per_article_total()')
data_now = views_per_article_total.loc[views_per_article_total["article_age_years"] >= 0]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(s=50)
g.map(plt.scatter, "article_age_years", "views_per_article", **kws);
g.map(plt.plot, "article_age_years", "views_per_article");
for ax in g.axes.flat: 
    ax.set_xlabel("article age (years)")
g.axes.flat[0].set_ylabel("views per article");


# **{{print figure_link("views-by-article")}}: Views by article curve, by OA type**

# <a id="section-11-4"></a>
# ### 11.4 Walk-through of convolution
# 

# This is a walk-through of convolution, as discussed in [Section 4.3.3](#section-4-3-3).  For each OA type:
# 
# ```
#  views in a given year = convolution(articles by age for that year, 
#                                      views/article by age)     
# ```
# 
# where [convolution](https://en.wikipedia.org/wiki/Convolution) is the standard mathematical operation of modifying a signal by another signal, by integrating the product of the two curves after one is reversed and shifted. 

# As an example, an estimate of total number of views in 2022 comes from summing together the views in 2022 across all article ages. In other words, the green X in the graph below at year 2022 is the area under the green curve in the row above -- the sum of all views to green OA of age 0, age 1, age 2, age 3, etc.
# We then did this for all years.
# 
# To show how we'll estimate views, we'll use 2022 as an example. We use the 2022 row from above, and the graph it by age of article (rather than year of publication). This flips the direction of the x axis. In this graph to make the next steps more clear we also use a shared y axis across all OA types.
# 

# In[88]:


register_new_figure('num-articles-2022');


# In[89]:



my_year = 2022
get_ipython().magic(u'cache papers_by_availability_year_total_2022 = get_papers_by_availability_year_total(my_year)')
data_now = papers_by_availability_year_total_2022.loc[papers_by_availability_year_total_2022["article_years_from_availability"] < 15]
data_now["publication_year"] = my_year - data_now["article_years_from_availability"]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict()
g.map(plt.bar, "article_years_from_availability", "num_articles", **kws);



# **{{print figure_link("num-articles-2022")}}: Predicted number of articles by OA type by article age, at the year of observation 2022**

# Next, we'll use the signal we calculated in the section "How often does someone want to access a paper, given its age and OA type", which shows the number of views per article someone made in 2019. An assumption in our model is that this views-per-article probability stays the same across time, so we assume that it applies to 2022 as well.
# 

# In[90]:


register_new_figure("views-per-article2");


# In[91]:



get_ipython().magic(u'cache views_per_article_total = get_views_per_article_total()')
data_now = views_per_article_total.loc[views_per_article_total["article_age_years"] >= 0]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(s=50)
g.map(plt.scatter, "article_age_years", "views_per_article", **kws);
g.map(plt.plot, "article_age_years", "views_per_article");


# **{{print figure_link("views-per-article2")}}: Views per article by OA type by article age**

# Now we multiply these two signals together. We multiply them in a similar way that we divided signals in an earlier step -- we take each OA type in turn, and then take each age bin in turn. So the green OA point at 0 years in the graph below comes by multiplying the number of estimated articles in 2022 that are available as green OA and 0 years old by the number of "views-per-article" we calculated for green OA for articles that are 0 years old. We then do that calculation for every age bin, for every OA type, and get the graph below:
# 

# In[92]:


register_new_figure("views-by-article-year-2022");


# In[93]:



get_ipython().magic(u'cache predicted_views_by_pubdate_total = get_predicted_views_by_pubdate_total(my_year)')
data_now = predicted_views_by_pubdate_total.loc[predicted_views_by_pubdate_total["article_age_years"] < 15]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(alpha=0.25, linewidth=8)
g.map(plt.plot, "article_age_years", "views", **kws);
# kws = dict(alpha=1, linewidth=5, linestyle='dashed')
# g.map(plt.plot, "article_age_years", "views", **kws);
for ax in g.axes[0]:
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.0f}'.format(y/(1000*1000.0))))
    ax.set_ylabel("views (millions)")


# **{{print figure_link("views-by-article-year-2022")}}: Views by OA type by article age for the observation year 2022**

# In[ ]:





# In[94]:


register_new_figure("views-small");


# In[95]:


observation_year = 2025
get_ipython().magic(u'cache predicted_views_total = get_predicted_views_total(observation_year)')

data_now = predicted_views_total.loc[predicted_views_total["observation_year"] >= 2010]
g = sns.FacetGrid(data_now, col="graph_type", hue="graph_type", col_order=graph_type_order, hue_order=graph_type_order, palette=my_cmap_graph_type)
kws = dict(marker="x", s=70)
g.map(plt.scatter, "observation_year", "views", **kws);
for ax in g.axes.flat: 
    ax.set_xlabel("view year")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '{0:,.0f}'.format(y/(1000*1000.0))))
g.axes.flat[0].set_ylabel("views (millions)");


# **{{print figure_link("views-small")}}: Views** Views by year.

# This gives us views for each year, 2010 to 2025, by OA type. The following graph is the same as the previous one, but without shared y axis so we can better see the relative trends. 
# 

# In[96]:


register_new_figure("views-large");


# In[97]:


fig, axes = plt.subplots(1, len(graph_type_order), figsize=(13, 3), sharex=True, sharey=False)
axes_flatten = axes.flatten()
plt.tight_layout(pad=0, w_pad=2, h_pad=1)
plt.subplots_adjust(hspace=1)
prediction_of_views = pd.DataFrame()
for i, graph_type in enumerate(graph_type_order):
    predicted_views_this_graph = predicted_views_total.loc[predicted_views_total.graph_type==graph_type]
    new_data = graph_views(graph_type, data=predicted_views_this_graph, ax=axes_flatten[i])
    new_data["graph_type"] = graph_type
    prediction_of_views = prediction_of_views.append(new_data)


# **{{print figure_link("views-large")}}: Views by year, full scale**

# <a id="section-11-5"></a>
# ### 11.5 Number of articles by OA type, by date of publication
# 

# In[98]:


register_new_figure("articles_by_simple_colors");
articles_by_simple_colors_df = articles_by_color_by_year.copy()
articles_by_simple_colors_df = articles_by_simple_colors_df.rename(
    columns={"published_year": "x",
             "num_articles": "y",
             "oa_status": "color"        
})


# You may be wondering what the data would look like if we ignored the idea of year of observation and simply plotted articles by year of publication, categorized by OA type as we measure it today.  We present this data in {{ print figure_link("articles_by_simple_colors") }}.

# In[99]:


plot_area_and_proportion(articles_by_simple_colors_df, 
                         "simple", 
                         1950, 2018, 2018,
                         fancy=None);


# **{{print figure_link("articles_by_simple_colors")}}: Articles by OA type, by year of publication. OA type as of October 2019.**

# The early years of {{ print figure_link("articles_by_simple_colors") }} is similar to Figure 2 in Piwowar et al. (2018), with one notable difference: some of what was considered Bronze OA (and to a lesser extent hybrid OA) in Piwowar et al. (2018) is classified as Gold OA in the current analysis. This is due to an improvement in Unpaywall's algorithms. Originally, Unpaywall used the Directory of Open Access Journals (DOAJ) as the sole arbiter of whether a journal was "fully-OA." Unpaywall still uses DOAJ in this way, but it now also adds an empirical check for OA journals (if 100% of a journal's articles are OA, it is listed as an OA journal). This results in a more comprehensive and accurate list of fully-OA journals, which in turn moves some articles into Gold from Hybrid and Bronze. We've made this comprehensive Gold OA journal list available: see [Section 7.1](#section-7-1).

# We can see a visible decrease in the proportion of OA (particularly Green) in the most recent publication years.  **This change in OA proportions is because many articles published in 2018 are still under embargo at the time of this analysis (October 2019), so they are considered "Closed" in this graph even though they may ultimately become Green OA or Bronze OA.**  More on this in [Section 4.1](#section-4-1).

# In[100]:


register_new_figure("articles_by_simple_colors_cumulative");


# A cumulative view of {{ print figure_link("articles_by_simple_colors") }} is shown in {{ print figure_link("articles_by_simple_colors_cumulative") }}.
# 

# In[101]:


plot_area_and_proportion(articles_by_simple_colors_df, 
                         "simple", 
                         1950, 2018, 2018,
                         fancy="cumulative");


# **{{print figure_link("articles_by_simple_colors_cumulative")}}: Cumulative articles (total articles extant in the world) by OA type, by year of publication. OA type as of October 2019.**

# In[ ]:




