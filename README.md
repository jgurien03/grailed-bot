A simple discord bot fork of my own Grailed scraper which you can find at [https://github.com/jgurien03/Grailed-Scraper](https://github.com/jgurien03/Grailed-Scraper). Currently, it is not hosted, so you can host it on your own computer if you'd like, just make sure to use your own bot token. Results may be affected by factors such as processing power and WiFi, as this is a web-based crawler that will yield different results based on performance.

**COMMANDS:**
- !search: starts the search for a brand. must supply email and password, as well as either 'sold' or 'current' as the response category
- !analyze: produces unique options that can display the data for the user, such as graphs, predictions, and the raw dataframe file
- !shutdown: shuts down the bot; only available for the owner (me) at the moment

**HOW TO RUN:** \n
Using the commands above, type them (in that order) to receive results. Once the search stops, you will receive a notification saying 'search complete.' The bot may go off for a short period of time, and once it comes back online, you can use the '!analyze' command. Once this is complete, you may search again.

**TO DO:**
- Get hosted
- Fix infinite scroll timeout
- Update model's train features to yield lower validation loss
- Include more dataframe options
