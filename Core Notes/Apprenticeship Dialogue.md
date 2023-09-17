
Hey there Blueprint team, my name is Madison Evans and today I will be walking you through the code I wrote for the Apprenticeship coding assessment. And so without any further ado, let’s just jump into it.

## Requirements 

I will first start off by taking a quick look at the requirements, and then showing the frontend in action, just to provide full context before jumping into the code

As we can see by the readme, the main project requirements were to

- Create Card components with the relevant information
- Use the data in the fake dataset to populate the cards
- Create a sorting functionality that sorts by ascending and descending order
- And have the cards be searchable

And with that, there are some challenges that have to be accounted for, namely the difference in naming conventions for  some of the platforms, as well as merging the google analytics data appropriately. All of which I will show my solutions for when we jump into the code

## Quick Demo

- Show the sorting
- Show the search
- Make a note about some of the items not having google analytics information and how we handled that
- Comment on the styling and show the responsiveness

## Code Demo

### File Structure and Config

Now let’s jump into the code. First I want to draw attention to my file structure and configuration settings. Even for an application as straight forward as this one, I still like to modularize and refactor in ways that bring the most amount of clarity and intention to my code. So in order to accomplish that, I’ve made 2 directories: one for my components, and the other for a custom hook that I built that handles the data fetching and data merging logic.

Additionally, I’ve updated the tailwind config file in order to introduce some level of theme for consistency. This made designing the UI incredibly fast and streamline because a lot of the interface considerations were thought about and made up front.

Taking a deeper look into the components, let’s start with the card component:

- The card component has a subcomponent called DataRow which holds a label and value. This corresponds to the data attribute of an ad (such as campaign name or cost), and the respective value for that instance. To protect against non-zero falsy values, I’ve placed this line of code to conditionally render alternative outputs

- This of course is all dynamically rendered based on the props that are passed in

Zooming out, we’ll next take a quick look at the CardList component, which is the container for the Card component.

- This is a pretty simple component which I decided to refactor out in order to be able to have more focused attention to the responsive styling.

- I also noticed that ads didn’t have a unique identifier such as an ID. Or rather, the unique identifier was a composite identifier based on the combination of campaign, adset, and creative. So I concatenated these together to form the key for the mapping function. I chose this rather than just using the default “I” index variable because React sometimes acts a bit wonky when you use that variable instead of a unique identifier

And last but not least, we have the panel component:

- This is where the search and filtering functionality resides

- As you can see, I’m using the inversion of control principle by passing some setters in as props. These setters take the selection input, and typed input as arguments respectively.

- The logic for these occurs above in the App.js file (explain)

### CUSTOM HOOK

- The final piece of code that glues all of the functionality together is a custom hook I made called “useDataServices”

- This hook provides functionality for fetching the data from the database, standardizing the parameter names of the ads, and merging the google analytics data into a new data structure to be consumed by the client

- While this application is quite simple, I still like to incorporate this MVC architecture just to keep things modular. If this was a much bigger project with a higher number of functional requirements, I think this design strategy would really come in handy

#### useDataServices
- starting with the useDataService function, this is the main function call that uses useState and useEffect hooks to fetch all the raw data from the database and then pass it to the helper functions for merging and normalizing data 
- Incorporating the environment variable for fetching from the api probably wasn't required for this, but it's good practice to do anyways so I incorporated it here 

#### standardizeAds(ads, platform)
- This function takes an array of ad objects (ads) and a platform name (platform) as arguments. It returns a new array of ad objects where each object is standardized to have a consistent set of keys.

- Different ad platforms might use different keys for similar data. For example, one platform might use campaign_name while another uses utm_campaign. The function standardizes these keys by using the JavaScript OR (||) operator to pick the first available key-value.

- It also adds a platform key to each ad object, which is useful for identifying the source platform of each ad.

- This function essentially harmonizes the data schema across different platforms, making it easier to aggregate or analyze the data later.

mergeAnalytics(ads, analytics)

- This function takes two arrays: ads, which contains standardized ad data, and analytics, which contains Google Analytics data.

- It first creates an object (aggregatedAnalytics) to hold the sum of results from Google Analytics, keyed by a combination of utm_campaign, utm_medium, and utm_content. (comment about repeat values)

- For each analytic data point, it forms a unique key and aggregates the results under this key.

- It then iterates through the ads array and looks for matching analytics data using a similar key formation strategy. If it finds a match, it adds the results to the ad object.

- No Data Count: If no matching analytics data is found, it increments a noDataCount variable and sets the results to "No Data".

- Finally, it logs the number of ads for which no analytics data was found. This wasn’t technically a requirement for this assessment, but I thought it was a nice to have

- The function returns a new array of ad objects that include the merged analytics data, providing a comprehensive view of each ad's performance.

- Both functions are crucial for data normalization and enrichment, enabling more straightforward and accurate downstream analysis or rendering.

## Conclusion 

That’s pretty much it in a nutshell. If you would like to take a more in depth look at my code and perhaps play around with it a bit, I’ve sent the team a link to the public github repo. Thanks so much for your time and consideration as I walked you through my project, and I look forward to hearing feedback from the team soon!