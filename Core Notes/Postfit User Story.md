
## Scenario 

I am a `28` year old `male` who weighs `180 lbs (~81.65kg)` and over the course of `30` days, I would like to lose `5 lbs (2.268 kg)`. I would like to keep a diet that provides me with an average caloric intake of `2500` calories/day. 

1. **Weight Loss Goal:** 2.27 kg
2. **Time:** 30 days
3. **Caloric Intake:** 2500 calories/day
4. **Weight:** 81.65 kg
5. **Height:** 180 cm
6. **Age:** 28 years
7. **Sex:** Male (C = 5)

We can determine the average Physical Activity Level with the [Mifflin-St Jeor](https://reference.medscape.com/calculator/846/mifflin-st-jeor-equation) equation. 

$$ W_{\text{loss}} = \frac{1}{C_{\text{to\_kg}}} \times \left[ \text{Cal}_{\text{intake}} - \left(10 \times W_{\text{start}} + 6.25 \times H - 5 \times A + G\right) \times \text{ave\_PAL} \right] \times T $$

Where:
- $( W_{\text{loss}} )$ = Weight loss goal (in kg)
- $( C_{\text{to\_kg}} )$ = Caloric equivalent of 1 kg weight loss (typically 7700 calories/kg)
- $( \text{Cal}_{\text{intake}} )$ = Average daily caloric intake
- $( W_{\text{start}} )$ = Starting weight (in kg)
- $( H )$ = Height (in cm)
- $( A )$ = Age (in years)
- $( G )$ = Gender factor (e.g., 5 for men, -161 for women)
- $( \text{ave\_PAL} )$ = Average Physical Activity Level 
- $( T )$ = Time period (in days)

>plugging in the values and solving for $\text{ave\_PAL}$ gives us a value of `1.50`. 

### User Story 

>Generate a Weekly Workout Plan to Achieve Target PAL

**As** a Postfit app user...
**I want** the application to create a weekly workout plan for me... 
**So that** I can achieve an average daily PAL of `1.5` while maintaining a daily caloric intake of `2500`.

#### Acceptance Criteria:

1. **Profile Info Verification:**  
When I log into the Postfit app,  
The application should already have my current weight (`81.65`kg), height (`180`cm), age (`28`years), and gender (`male`) stored.

2. **Goal Setting:**  
I should be able to specify that I want to lose `2.27`kg over a `30` day period.

3. **Dietary Input:**  
When prompted for my average daily caloric intake,  
I should be able to input that I plan to consume `2500` calories per day.

4. **Workout Plan Generation:**  
Once my goals and dietary inputs are set,  
The app should generate a balanced weekly workout plan tailored for me.

5. **PAL Achievement:**  
The suggested workouts in the plan should be crafted in such a way that,  
When combined with my resting metabolic rate and daily activities,  
My average PAL over the week should come out to approximately `1.5`.

6. **Plan Flexibility:**  
I should be able to modify the workout plan if a certain day doesn't suit my schedule,  
With the app recalibrating subsequent days to ensure I stay on track to achieve my target PAL.

7. **Progress Tracking:**  
The app should allow me to log my workouts and track my progress,  
Offering feedback on whether I'm on track to meet my `1.5` PAL goal by the end of the week.

---
## Calculating PAL 

[Physical Activity Level (PAL)](https://en.wikipedia.org/wiki/Physical_activity_level) is a multiplier used to calculate TDEE based on resting metabolic rate. It varies, but can be approximated based on how active a user is during the day. One way to correlate a specific workout is to use the [Metabolic Equivalent of Task (MET)](https://nutrium.com/blog/how-to-calculate-the-pal-with-mets-in-nutrium/#:~:text=The%20Physical%20Activity%20Level%20(or,equation%20PAL%20%3D%20TEE%2FBMR.). There are many resources online that can be used to find the MET for a particular workout(s). The formula linking PAL to MET is as follows: 
$$ \text{PAL} = \frac{\text{Total MET-hours for the day}}{\text{Hours in a day}} $$
...so if I did a workout for 1 hour that had a MET value of 10, then my PAL value for that day is as follows 
$$\text{PAL} = \frac{10*\text{1 hour} + 1*\text{23 hours}}{\text{24 hours}} = \frac{33}{24} = 1.375$$

---
## Chat GPT Integration

If we have a target average PAL level, and some notion of how frequently we want to work out in a given week, we can utilize chat gpt to do the rest by requesting a workout based on desired frequency and desired MET/PAL value

### Example 

> Prompt: 

I have a target of `1.5` average daily PAL. Provide for me a workout plan for the week such that the workouts have an MET value that will bring me to an daily average PAL value of `1.5`. Assume that the sedentary baseline PAL is `1.2`. Workouts should be no longer than `90` min.


> Answer: 

Each `90` minute workout should have an intensity of around `6.72` MET to reach your target.

**Monday: Cardio + Strength Circuit (6.7 MET)**

*Equipment*: Treadmill, Dumbbells

* Warm-up: 
  * 10 minutes on the treadmill at a brisk walk.

* Main Set:
  * 5 minutes of jogging on the treadmill.
  * 3 x 10 dumbbell squats.
  * 3 x 10 dumbbell lunges (each leg).
  * 3 x 10 dumbbell chest press.
  * 5 minutes of jogging on the treadmill.

* Cool Down:
  * 10 minutes of walking on the treadmill, gradually decreasing the speed.

---

**Tuesday: Interval Training on Treadmill (6.7 MET)**

* Warm-up: 
  * 10 minutes of brisk walking.

* Main Set:
  * 1 minute of running at a fast pace.
  * 1 minute of walking.
  * Repeat this 1-minute fast run, 1-minute walk cycle for 30 minutes.

* Cool Down:
  * 10 minutes of easy walking.

---

**Thursday: Full Body Strength Training (6.7 MET)**

*Equipment*: Dumbbells, Bench

* Warm-up:
  * 10 minutes of dynamic stretches and light dumbbell exercises.

* Main Set:
  * 3 x 10 dumbbell squats.
  * 3 x 10 bench press.
  * 3 x 10 dumbbell rows (each arm).
  * 3 x 10 seated dumbbell shoulder press.
  * 3 x 10 dumbbell bicep curls.
  * 3 x 10 dumbbell tricep extensions.

* Cool Down:
  * 10 minutes of stretching focusing on the major muscle groups you worked.

---

**Friday: Elliptical and Core Workout (6.7 MET)**

*Equipment*: Elliptical machine, Mat

* Warm-up:
  * 10 minutes easy pace on the elliptical.

* Main Set:
  * 10 minutes on the elliptical, increasing the resistance every 2 minutes.
  * 3 x 15 sit-ups.
  * 3 x 20 Russian twists (using a light dumbbell if desired).
  * 3 x 15 leg raises.
  * 10 minutes on the elliptical, decreasing the resistance every 2 minutes.

* Cool Down:
  * 5 minutes of easy pace on the elliptical followed by stretching.

---

**Sunday: Mixed Cardio (6.7 MET)**

*Equipment*: Stationary bike, Jump rope

* Warm-up:
  * 10 minutes of easy cycling.

* Main Set:
  * 10 minutes of moderate-paced cycling.
  * 5 minutes of jump rope.
  * 10 minutes of moderate-paced cycling.
  * 5 minutes of jump rope.

* Cool Down:
  * 5 minutes of slow-paced cycling followed by stretching.


---


![[Screen Shot 2023-11-07 at 10.21.59 AM.png]]


---

![[Screen Shot 2023-11-07 at 10.23.02 AM.png]]