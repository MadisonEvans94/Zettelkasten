#seed 
###### upstream: [[NoSQL]]

**MongoDB** is a source-available cross-platform document-oriented database program. It is classified as a NoSQL database program, which means it does not use the traditional table-based relational database structure, but rather a variety of data models, including key-value, document, columnar, and graph formats.

MongoDB is particularly known for its use of the document data model. In MongoDB, data is stored in flexible, JSON-like documents. This means fields can vary from document to document (there is no set schema enforced by the database itself), and data structure can be changed over time. This data model makes it easy to store and combine data of any structure, without disrupting applications that are already running.

*Key features of MongoDB include:*

-   **Flexible Data Format**: Data in MongoDB is stored in BSON, a binary representation of JSON, allowing for diverse data types to be stored.
    
-   **Scaling Out**: MongoDB is built for horizontal scale-out. As your data grows, you can add more machines to handle the increased load.
    
-   **Support for Aggregation**: MongoDB provides a powerful aggregation framework to process data and return computed results.
    
-   **Indexing**: You can index any attribute in a document to improve search performance.
    
-   **Replication & High Availability**: MongoDB maintains multiple copies of data using a process called replication. This enhances the availability of data and ensures the durability of the system.
    
-   **Sharding**: This is a method for storing data across multiple machines. MongoDB uses sharding to support deployments with very large data sets and high-throughput operations.
    
-   **Text Search**: MongoDB supports querying for text in multiple languages.
    

MongoDB is popular in many modern web applications as it pairs well with other technologies in the JavaScript ecosystem (like Node.js, Express.js, and frontend frameworks/libraries like React, Angular, Vue.js), often referred to as the ME(N/V/R/A)N stack.