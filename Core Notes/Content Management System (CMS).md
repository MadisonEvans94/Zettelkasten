
#incubator 

###### North: [[Web Development]]
### What 
- A software that allows non-technical people to build websites quickly and easily
- If the service is hosted for you (like Wordpress) then it is considered a [[SaaS]]
- Common Examples: 
	- Wordpress
	- Joomia
	- Drupal 
	- Magento 
	- Shopify
	- Squarespace
	- Wix
	- Contentful

### Why 
- Allows non-technical people to build websites easily 
### How
- CMS platforms have two main parts: 
	- **Content Management Application**: The user-facing interface where users can add, modify, and delete contentn without needing to know html or other programming languages (WYSIWYG)
	- **Content Delivery Application**: The part of the CMS that compiles the content and assets and then updates the website. It's responsible for storing the content (usually in a database) and delivering it to the user's browser when a page is requested
- A Content Delivery Application of a CMS is more akin to ther [[Server-side Rendering (SSR)]] process
	1. User makes a request to a webpage 
	2. CDA retrieves the necessary content from the database
	3. CDA combines content with appropriate templates to construct the html 
	4. HTML is sent to client's browser
