#seed 
upstream:

---

**video links**: 

---

**brain dump**: 

--- 

Proper spacing can elevate your mobile web application design to the next level, providing a more visually appealing and user-friendly experience. Here are some essential guidelines to follow when crafting your mobile app design system.

## Spacing

Spacing is a critical aspect in mobile design to avoid a cluttered look and maintain user focus on the most important elements. Here's how to approach it:

### Margins and Padding

#### Screen Margins
Typically, margins around the edge of the screen should be **16px** to **20px** on mobile devices. This creates a comfortable space between the content and the edge of the device, making it easier for users to focus on the content.

#### Between Elements
A good rule of thumb is to keep **8px** of vertical space between related components, and **16px** between unrelated components. Horizontally, keep at least **8px** between elements, unless grouping related elements closer together for a clear visual relationship.

### Grid System

Utilize a grid system to maintain a consistent layout. In a **12-column** grid, aim for **8px gutter** width (the space between columns).

### Touch Targets

Buttons, links, and other touch elements should be at least **48px by 48px** in size to be comfortably tapped with a finger, following the guidelines by the WCAG (Web Content Accessibility Guidelines).

## Font Size and Typography

The typography in your app significantly affects its readability and overall UX.

### Body Text

The optimal size for body text on mobile is typically between 14px and 16px. Anything smaller might compromise legibility.

### Headings

Heading sizes should be proportionally larger than body text, typically starting from 20px for H1, decreasing by 2px for each subsequent heading level (H2, H3, etc).

### Line Spacing

Line spacing (leading) plays a significant role in readability. For body text, aim for 120-145% of the font size. For example, if your body text is 16px, the line spacing should be around 19-23px.

### Letter Spacing

Typically, body text doesn't need extra letter spacing. For uppercase text, such as labels or buttons, consider adding 5-10% letter spacing to enhance readability.

## Layout Design

How you structure your design can significantly impact the user's ability to navigate and use your app.

### Consistency

Consistency in spacing, font size, color, and other design elements across your app makes it easier for users to understand and navigate it.

### Responsiveness

Ensure your design adapts to different screen sizes, keeping a consistent look and feel. 

### Hierarchy

Create a visual hierarchy to guide users' attention to the most important elements first. Larger or bolder fonts, contrasting colors, and placement on the screen can help establish this hierarchy.

### Grouping

Group related elements together to make it easier for users to understand their relationships. Use spacing and dividers to distinguish these groups visually.

## Navbar and Header Spacing

### Navbar Size

The height of the navbar on mobile is typically between 50px and 60px. This allows for easy readability and clickability of the buttons within it, without taking up too much screen space.

### Vertical Space Below Navbar

The vertical space between the bottom of the navbar and the first header or component on the page should ideally be 16px to 24px. This spacing can help separate the static navigation from the dynamic content, allowing the user to focus on the content.

> in figma, how do I easily set the space between two elements? For example, I have a navbar component, and beneath that, I have an h1 header. I would like to specify 24px of space between the bottom of the navbar and the top of the header. How do I do this?

*Setting the space between two elements in Figma is a relatively straightforward process.* Here's how you can specify a 24px space between your navbar component and an H1 header:

1. **Select the H1 Header**: Click on the text box or frame of your H1 header.

2. **Positioning**: In the right-side design panel, you'll see a section named "Layout". Here you can set the position of your elements in the X and Y coordinates. The Y-coordinate determines the vertical positioning of your element.

3. **Adjust the Y-Coordinate**: Increase or decrease the Y-coordinate value to create the desired space between the navbar and the header. As you adjust the value, you'll see the header move up or down.

4. **Measure the Distance**: To ensure there's precisely a 24px gap, use Figma's built-in measuring tool. Press "Option" (on macOS) or "Alt" (on Windows) on your keyboard, then click and drag from the bottom edge of your navbar to the top edge of your header. A red line with a number will appear, indicating the distance in pixels. Adjust the Y-coordinate until this reads 24.

Remember, the coordinates relate to the parent frame, so ensure both your navbar and H1 are in the same frame (like the frame of the device screen) when you are doing these adjustments. This way, you'll be sure they are correctly positioned relative to each other.

For future designs, consider using the "Auto Layout" feature that lets you set automatic spacing between elements within a frame. This way, when you add or adjust elements, Figma automatically maintains the specified space between them.

## Form Layout

### Form Fields

For a form asking for name, email, category, and message body, start with the least personal information and move to the most personal. 

- **Name Field**: Position this at the top. 
- **Email Field**: This should be below the Name field.
- **Category Dropdown/Selector**: Position this below the Email field.
- **Message Body**: This should be the largest and the last field, positioned below the Category.

### Form Spacing

Vertical spacing between each input field should ideally be around 16px to 24px to provide a clear separation between fields and make the form easier to navigate. 

## Testing

Test your design on various devices and screen sizes to ensure your spacing, typography, and layout work well in all scenarios. User testing will also provide valuable feedback to further refine your design.

These guidelines are starting points and might need adjusting based on your specific design requirements and user feedback. Always design with your end user in mind to create an enjoyable and easy-to-navigate mobile app.