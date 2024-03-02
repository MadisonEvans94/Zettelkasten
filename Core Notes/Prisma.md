#seed 
upstream:

---

**links**: 

brain dump: 

---

## What is Prisma? 

Prisma ORM is an [open-source](https://www.prisma.io/docs/orm/overview/introduction/what-is-prisma#:~:text=Prisma%20ORM%20is,in%20your%20database.) next-generation ORM. It consists of the following parts:

- **Prisma Client**: Auto-generated and type-safe query builder for Node.js & TypeScript
- **Prisma Migrate**: Migration system
- **Prisma Studio**: GUI to view and edit data in your database.

Prisma Client can be used in _any_ Node.js (supported versions) or TypeScript backend application (including serverless applications and microservices). This can be a [REST API](https://www.prisma.io/docs/orm/overview/prisma-in-your-stack/rest), a [GraphQL API](https://www.prisma.io/docs/orm/overview/prisma-in-your-stack/graphql), a gRPC API, or anything else that needs a database.

## How Does Prisma Work? 
### The Prisma schema

Every project that uses a tool from the Prisma ORM toolkit starts with a [Prisma schema file](https://www.prisma.io/docs/orm/prisma-schema). The Prisma schema allows developers to define their _application models_ in an intuitive data modeling language. It also contains the connection to a database and defines a _generator_:

```javascript
datasource db { 
	provider = "postgresql" 
	url = env("DATABASE_URL") 
} 

generator client { 
	provider = "prisma-client-js" 
} 

model Post { 
	id Int @id @default(autoincrement()) 
	title String 
	content String? 
	published Boolean @default(false) 
	author User? @relation(fields: [authorId], references: [id]) 
	authorId Int? 
} 
	
model User { 
	id Int @id @default(autoincrement()) 
	email String @unique 
	name String? 
	posts Post[] 
}
```

## Prisma Client 
### Accessing your database with Prisma Client

#### Generating Prisma Client

The first step when using Prisma Client is installing the `@prisma/client` npm package:

```bash
npm install @prisma/client   
```

Installing the `@prisma/client` package invokes the `prisma generate` command, which reads your Prisma schema and _generates_ Prisma Client code. The code is [generated into the `node_modules/.prisma/client` folder by default](https://www.prisma.io/docs/orm/prisma-client/setup-and-configuration/generating-prisma-client#the-prismaclient-npm-package).

After you change your data model, you'll need to manually re-generate Prisma Client to ensure the code inside `node_modules/.prisma/client` gets updated:

```
prisma generate   
```

#### Using Prisma Client to send queries to your database

Once Prisma Client has been generated, you can import it in your code and send queries to your database. This is what the setup code looks like.

```javascript 
import { PrismaClient } from '@prisma/client' 
const prisma = new PrismaClient()
```

Now you can start sending queries via the generated Prisma Client API, here are a few sample queries. Note that all Prisma Client queries return _plain old JavaScript objects_.

Learn more about the available operations in the [Prisma Client API reference](https://www.prisma.io/docs/orm/prisma-client).

**Retrieve all `User` records from the database**

```typescript
// Run inside `async` function  
const allUsers = await prisma.user.findMany()   
```

**Include the `posts` relation on each returned `User` object**

```typescript
// Run inside `async` function  
const allUsers = await prisma.user.findMany({    include: { posts: true },  })   
```

**Filter all `Post` records that contain `"prisma"`**

```typescript
// Run inside `async` function  
const filteredPosts = await prisma.post.findMany({    
	where: {      
		OR: [        
			{ title: { contains: 'prisma' } },        
			{ content: { contains: 'prisma' } },      
		],    
	},  
})   
```

**Create a new `User` and a new `Post` record in the same query**

// Run inside `async` function  const user = await prisma.user.create({    data: {      name: 'Alice',      email: 'alice@prisma.io',      posts: {        create: { title: 'Join us for Prisma Day 2020' },      },    },  })   ``

**Update an existing `Post` record**

```typescript 
// Run inside `async` function  
const post = await prisma.post.update({    
	where: { id: 42 },    
	data: { published: true },  
})
```   





