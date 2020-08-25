# Analysing Ticket to Ride Maps

The goal of this project is to analyse some of the statistics and properties from maps in the boardgame [Ticket to Ride](https://www.daysofwonder.com/tickettoride/en/).

This is a hobby project that combines two of my interests: board games, and programming, with a taste of data science.

If you have any suggestions or would like to be involved please get in touch!

![The US map from the original Ticket to Ride](data/USA/USA_map.jpg)

## What do we analyse?

The core structure of every map in ticket to ride (with some small changes for some expansion maps) is a network of cities connected by railway tracks. In the game, players place coloured train counters on the tracks to connect cities, making their own personal network of connected locations. They get points for laying tracks as well as for fulfilling unique route objectives, which require them to create an unbroken link of tracks between two particular cities on the map. The harder the route, the more points it is worth.

I am interested to learn statistically which specific connections are likely to be the most sought after during the game. Are all connections created equal, or are some more likely to be occupied by the end of the game than others? This might give me some insight when playing the game of which areas to avoid, or which ones to ensure I build first before they get built by other players.

## Results

![Histogram of results](plots/heat_map_10_shortest.png)

## Future plans

Building on this, I would like to develop a tool for calculating for each player at any point during the game what the most valuable parts of unbuilt track are. This is not because I want to be as mean as I can (although this tool might allow you to do that) but rather just because I am curious. Even further down the road I might like to build an AI that uses this kind of value metric to make decisions about where to place track. 

## Overview

So far I have only analysed the US map. However, the scripts are easily editable for use with any map. The only time-consuming part of analysing a map is manually telling the scripts where the stations and the connections are, although even this I have automated slightly. 

### File structure:

* /data/ - contains the raw and processed map data
    * /USA/ - data for USA map; when adding other maps include these in different subdirectories
        * USA_map.jpg - original jpg image of the game board
        * USA_map_with_stations.svg - game board image with annotated circles added at each station, labeled with the station name
        * routes.csv - list of connections between stations, including length and colour
        * tickets.csv - list of ticket objective cards, including which cities to connect and reward in points
        * station_locations.json - dictionary of locations of stations; generated by /bin/get_stations.ipynb
* /bin/ - contains the executable files, written as Jupyter notebooks but easily convertible to Python scripts
    * get_stations.ipynb - extract station locations and names, then save as a dictionary in /data/USA/station_locations.json




