#incubator 
###### North: [[Software Development]]
###### South: 



### What
Best practices for strucutring typescript code for improved reusability and clarity

### Why
They provide standard structured approaches and solutions to common problems that have stood the test of time

### How 
There's a list of common patterns and their solutions that must be memorized 



#### Component (Composite) 
- The lego approach. Essentially, you build bigger components out of smaller components
- The entire philosophy behind [[React]] is built off of the idea of compoents fitting in to each other 
- The Layout Component is a tangible React example 
```ts
interface CarComponent {
	operation(): string; 
}

class Car implements CarComponent {
	protected components: CarComponents[] = []; 

	public add(component: CarComponent): void {
		this.components.push(component); 
	}

	public operation(): string {
		return `Car with components: ${this.components.map(component => component.operation()).join(', ')}`; 
	}
}

class Engine implements CarComponent {
	public operation(): string {
		return 'Engine'; 
	}
}

class Wheel implements CarComponent {
	public operation(): string {
		return 'Wheel'
	}
}
```
^ in this example, the Car component is the interface, or **blueprint**. It is a contract that every child must follow. This is useful because it allows us to perform the [[Liskov Substitution Principle]]

#### Factory  
- As the name suggests, the factory pattern involves having a class dedicated to building a certain object when you need it 
- Instead of using the `new` operator for object instantiation, you're using the `.build` method of a factory object
- Decoupling: By using a factory, you are decoupling the object instantiation in case you need to modify how a certain object is built without introducing any breaking changes
```ts
class Car {
	constructor(public model: string){}
}

class CarFactory {
	static create(model: string) {
	return new Car(model)}
}

// instead of using "new Car()", we use the factory: 
let sedan = CarFactory.create('Sedan'); 
let suv = CarFactory.create('SUV')
```
#### Decorator 
#### Strategy 
#### Observer
#### Command 
#### Singleton 