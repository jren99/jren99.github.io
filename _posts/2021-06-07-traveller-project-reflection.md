---
layout: post
title: Blog Post 4
---

What are three things you learned from the experience of completing your project? Data analysis techniques? Python packages? Git + GitHub? Etc? 

How will your experience completing this project will help you in your future studies or career? Please be as specific as possible.

For a math student who aspires to be a data scientist, opportunities to collaborate with others to complete a coding project are extremely valuable for me. When I was taking PIC 16B at UCLA, I was very lucky to meet [Ashley Lu]() and [Jingxuan Zhang]() because of our common interests in travelling, especially during this unusual circumstance of Covid 19, and we were able to learn, think, and design this project together, [LA Travel Planner](https://github.com/jren99/pic16b_project), which was not only a programming project that we are all really proud of but also a reflection of what we want to realize in real world using what we've learned in the classroom. I want to take this opportunity to share my experience with you in hopes of motivating you if you are struggling with your own project. 

### Project Overview

![png](/images/Home.png)

When it comes to traveling, sometimes it can be a struggle to plan out where you want to go, especially if you're going somewhere you've never been to before. Our project creates a travel planning tool that gives attraction, hotel, and food recommendations to LA tourists, and provides a detailed and personalized travel plan based on users' selections, including attractions to go for each day and a route recommendation. 

For example, under "Restaurant Recommendations", users can either enter the type of food they are interested to filter corresponding restaurants, or enter a number (for example, 3.5) to filter restaurants with ratings higher than the input number.

![png](/images/food.png)

For the "Plan Your Trip!" section, users can generate a customized travel plan based on their inputs.

![png](/images/plan.png)

After entering the places, duration, hotel, transportation information, the generator will generates a plan summary and maps for each day.

![png](/images/step2.png)

Users can click on each day to see the corresponding map.

![png](/images/step3.png)

For more details, please refer to the project [repo](https://github.com/jren99/pic16b_project). 

### Highlights 

This project has been a wild journey. We came up with a proposal that none of us knew how to make it come true but eventually achieved something that none of us ever thought we were able to. The first thing I'm the most proud of our project is the route generator. It was the first thing we accomplished in our project, which was very encouraging. At the beginning, we were only able to generate the shortest route between locations, but then we were able to add hotels and the transportation options to make our generator more dynamic. I also want to thank Ashley for making all those beautiful maps, which built the most important foundation of the whole project. 

The second thing that I'm also really proud of is our front-end. We managed to build a functioning webapp from scratch to demonstrate what we've accomplished for our project. I had some limited experience with html and css to build static website during hackathons but I had never done anything close to this. In order to connect to the database we scraped from [Tripadvisor](https://www.tripadvisor.com/) and the geomaps, we've tried to learn `PHP` first. Although we successfully connected to the database and it allows users to search for attractions, we had trouble to realize the route generator. We've also tried to learn `Django`, but it was too complicated to grasp within limited time. At the end, we learned to use `flask`, which is also python-based like `Django`, to build our final product. In my biased opinion, it is simple, clean, and beautiful. 

### Proposal v.s. Final Project

Of course, there were things we wanted to accomplish but weren't able to fully deliver. In the original proposal, we wanted to create such a travel planning tool for travellers to California, possibly to the entire United States, even to the whole world. Quite ambitious, right? However, because the scraping method wasn't efficient enough, we decided to focus on travellers to LA and created database for attractions, hotels, and restaurants in LA. Initially, we also wanted to generate attractions, hotels, and restaurants recommendations based on the city the users input. However, because we limited the city to Los Angeles, we decided to use keyword searching instead, which allows users to filter results based on types or ratings. However, I'm very pleased with what we've accomplished. 

Although we limited our users to a much smaller scope, we almost realized all functionalities in our originial proposal. We can successfully display the optimal routes for traveling between the sightseeing locations and the hotel location and the estimate travel time based on the transportation. We also implemented a webapp to integrate all functionalities. Overall, I think we've achieved what we planned at the beginning.

### Limitations 

Because of lack of experience, we had to self-learn most of the time. As a result, there are still a lot more to improve for our project. It I were to pick two things I want to improve upon, the first thing will be to find a better way to display the route visualizations. The users are limited to have at most 10 maps because we had to manually create corresponding links for each map, so we only wrote 10 links. However, if I were able to find a way to display all maps on the same page, then this limitation will be solved. The second thing I want to improve is to create a user database to allow users to create accounts and save travel plans they've created. 

### Things I've Learned 

1. `Github`
I 