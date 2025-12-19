# three-and-D-NBA-player-analysis (By Bhargav Ashok)


The term "Three and D" players has been frequently mentioned over the last two decades. 

A Three and D player is a guard or forward who can stretch the floor during an NBA matchup while possessing a special skillset of lockdown defense, and can get the team quick and easy points to secure their lead, shrink defects and shift momentum. 

With the game constantly changing, the role of a three-and-D player may not seem as important during heated exchanges, playoff runs, or even regular-season games. Is this true?

The question we are trying to answer today is: What statistics are important for a three-and-D player? And do they still make an impact? 

Well, to answer this question, we will run a random forest regression model to predict feature importance based on the typical makeup of a 3&D player (filtering by SG, SF, and PF, focusing on 3-point accuracy, Defensive stats, points per game, usage, true shooting percentage, etc). 

The model's ouput show us that 3&D players are still very much active in the NBA today. However, the focus has shifted to immediate contribution, as the number of games started has the most influence. 

Now, there are some caveats to these findings, as the dataset focuses on players from 1980 to the present who fit 3&D characteristics except star players at the SG, SF, and PF positions (outliers). The majority of the data is crowded around the line of best fit, and that is what we will focus on (though star players may/may not possess 3&D skillsets, our outliers assume that they do). 

The next metric below games started is points per game, followed by true shooting percentage. The shift in offensive focus in the modern 3-point era is significant for 3&D players, as scoring and keeping pace have become the name of the game. The first defensive stat we see is block percentage.

Block percentage, which usually increases as you get closer to the paint, requires strength to prevent box-outs and intuition/estimates from the 3&D player on when to execute (through experience and film).

So, do 3&D players still exist? Yes, they do, and they won't be going away anytime soon, though many players can possess qualities of 3&D today compared to when the archetype was first discovered in the late 2000s to early 2010s.

So, what does a 3&D player look like in 2025/26? A player who possesses a skillset with interior defense capabilities, who can swing the ball, and make an immediate impact. A 3&D player of the 2020s is your role player of the 2010s. 

# Metrics

<img width="1400" height="500" alt="Figure_12" src="https://github.com/user-attachments/assets/bc9ed959-a540-44df-b8c6-5f0d42db7a24" />

<img width="1400" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/00f657cb-6bf1-402f-8e08-4beb3e2b808e" />

# Resources
https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/

https://www.sportsvisio.com/stories/basketball-stats-explained#:~:text=VORP%20(Value%20over%20Replacement%20Player,prorate%20to%20a%20full%20season.

https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats?resource=download (DATASET IS HERE)

https://bleacherreport.com/articles/1040309-understanding-the-nba-explaining-advanced-defensive-stats-and-metrics

https://hackastat.eu/en/learn-a-stat-box-plus-minus-and-vorp/

https://www.basketball-reference.com/about/glossary.html

https://hoopstudent.com/basketball-3-and-d-player/




Hope you all enjoyed this analysis!

Tech Stack: Python (Random Forest Regressor + Claude 4.5 for semantic/syntax model debugging)
