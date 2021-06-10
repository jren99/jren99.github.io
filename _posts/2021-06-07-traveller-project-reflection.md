---
layout: post
title: Blog Post 4 - Project Reflection
---

For a math student who aspires to be a data scientist, opportunities to collaborate with others to complete a coding project are extremely valuable for me. When I was taking PIC 16B at UCLA, I was very lucky to meet [Ashley Lu](https://ashley-lu.github.io/reflection-blog-post/) and [Jingxuan Zhang](https://stancyzhang.github.io/Reflection-Blog-Post/) because of our common interests in travelling, especially during this unusual circumstance of Covid 19, and we were able to learn, think, and design this project together, [LA Travel Planner](https://github.com/jren99/pic16b_project), which was not only a programming project that we are all really proud of but also a reflection of what we want to realize in the real world using what we've learned in the classroom. I want to take this opportunity to share my experience with you in hopes of motivating you if you are struggling with your own project. 

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

This project has been a wild journey. We came up with a proposal that none of us knew how to make it come true but eventually achieved something that none of us ever thought we were able to. 

The first thing I'm the most proud of is the route generator. It was the first thing we accomplished in our project, which was very encouraging. At the beginning, we were only able to generate the shortest route between locations, but then we were able to add hotels and the transportation options to make our generator more dynamic. I want to thank Ashley for making all those beautiful maps, which built the most important foundation of the whole project. 

The second thing that I'm also really proud of is our front-end. We managed to build a functioning webapp from scratch to integrate what we've accomplished in our project. I had some limited experience with html and css to build static websites during Hackathons but I had never done anything close to this. In order to connect to the database we scraped from [Tripadvisor](https://www.tripadvisor.com/) and the geomaps, we've tried to learn `PHP` first. Although we successfully connected to the database and it allows users to search for attractions, we had trouble to realize the route generator. We've also tried to learn `Django`, but it was too complicated to grasp within limited time. At the end, we learned to use `flask`, which is also python-based like `Django`, to build our final product. In my biased opinion, it is simple, clean, and beautiful. 

### Proposal v.s. Final Project

Of course, there were things we wanted to accomplish but weren't able to fully deliver. In the original proposal, we wanted to create such a travel planning tool for travellers to California, possibly to the entire United States, even to the whole world. Quite ambitious, right? However, because the scraping method wasn't efficient enough, we decided to focus on travellers to LA and created database for attractions, hotels, and restaurants in LA. Initially, we also wanted to generate attractions, hotels, and restaurants recommendations based on the city the users input. However, because we limited the city to Los Angeles, we decided to use keyword searching instead, which allows users to filter results based on types or ratings. 

Although we limited our users to a much smaller scope, I'm very pleased with what we've accomplished. We almost realized all functionalities in our originial proposal. We can successfully display the optimal routes for traveling between the sightseeing locations and the hotel location and the estimate travel time based on the transportation. We also implemented a webapp to integrate all functionalities. Overall, I think we've achieved what we planned at the beginning.

### Limitations 

Because of lack of experience, we had to self-learn most of the time. As a result, there are still a lot more to improve for our project. If I were to pick two things I want to improve upon, the first thing will be to find a better way to display the route visualizations. The users are limited to have at most 10 maps because we had to manually create corresponding links for each map, so we only wrote 10 links. However, if I were able to find a way to display all maps on the same page, then this limitation will be solved. The second thing I want to improve is to create a user database to allow users to create accounts and save travel plans they've created. 

### Things I've Learned 

#### Github

I used Github for projects and Hackathons before, but I wasn't comfortable using it. Through this experience, I now truly understand the advantage and convenience of Github, especially for collaborations with others. Because we constantly use it to update our progress on the project, I've become more and more comfortable using Github. 

#### Web Development

Like I mentioned before, I had previous experiences building webpages during Hackathon, but only with front-end. This is the first time for me to build something complete and functional (instead of static). I was able to learn and understand more of the structure of front and back ends. It was also a great experience to learn something about `php`, even though we decided not to use it at the end. I learned how to write and strucutre `flask`. I had lots of fun working with Jingxuan on Zoom to figure out together how to make things work on flask.

#### Database Query

I've learned SQL query and practiced in different assignments, but it feels very different when it comes to a real project, because the dataset doesn't look as nice and somehow I just couldn't wrap my head around data query in web development at the beginning. It feels very straightforward now but I had a difficult time understanding how to take the user input as part of the query and display the output on the webpage. 

#### Stay in it when it gets tough. 

It does get exhausing when I get stuck, sometimes even for days. However, it's important to have faith in yourself and know that you will figure it out eventually. Don't quit. If you can't solve it and start feeling anxious, take a break. Play with your cat (or dog). Take a walk. Listen to Taylor Swift. Watch an epsiode of your favorite TV show. And when you feel better, come back to your computer to resume your work. 

### Lastly...

This project has been an amazing journey that builds my confidence in learning more about programming and confirms my passion in a career related to data and programming. I used to think I wasn't capable enough to code and get intimidated by programming sometimes, because C++ gave me headaches. The fact that we are able to complete this project helps me overcome my imposter syndrome and I feel much more confident now about coding. This project has definitely helped me to develop a growth mindset. Now I believe that even if I don't know something now, I'm able to learn it and will eventually grasp the concepts. I want to thank my professor Phil Chodrow for this great course and generous help he provided throughout the whole course. 

As for my passion in a career in data science, it started from my regression analysis and Python courses. Those experiences made me realize my enthusiasm about data science and Python is FUN, indeed! So I decided to take more python classes. This class is my first project-based class and my first hands-on experience of what a project in the real-world feels like, from coming up with the idea to make it come true. Although our project isn't directly related to data analysis and definitely much easier than simpler than real-world projects, I gained so much from it. 

- I learned about proper work pipeline and collaboration through Github, and small things like how to write a professional Readme file. 
- I also learned that it's okay not to know everything, because project is a learning process. I just need to calm down, decompose complex problems into smaller and simpler ones, and find solutions for each, either through asking specific problems on Google (or other platform) or seeking guidance from others. 
- More importantly, I truly enjoyed collaborating with my teamates on this project. Although there were times when I got stuck for days or ran into a deadend, they were always there to support me, so I want to thank Ashley and Jingxuan for this incredible journey. I still clearly remember those thrilling moments of a breakthrough, which filled me with satisfaction and sense of accomplishment. 

I feel I can actually create something meaningful with programming and those error-free lines of code make me smile. I can definitely say that this experience makes me want to work in data industry even more. 