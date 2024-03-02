#seed 
upstream:

---

**links**: 

---



## Setting Up Config 

> Here's an example of a tailwind config file that sets up for an entire design system 

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js,jsx,ts,tsx}"],
  theme: {
    extend: {
      // Custom Colors
      colors: {
        peach: '#fee9d1',
        primary: '#007bff',
        secondary: '#6c757d',
        success: '#28a745',
        info: '#17a2b8',
        warning: '#ffc107',
        danger: '#dc3545',
        light: '#f8f9fa',
        dark: '#343a40',
      },
      // Custom Font Families
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        serif: ['Merriweather', 'serif'],
      },
      // Custom Spacing
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },
      // Custom Border Radii
      borderRadius: {
        'default': '24px',
        'large': '12px',
      },
      // Custom Typography (FontSize, FontWeight, LineHeight)
      fontSize: {
        'xs': ['.75rem', '1.5'],
        'sm': ['.875rem', '1.5'],
        'base': ['1rem', '1.5'],
        'lg': ['1.125rem', '1.5'],
        'xl': ['1.25rem', '1.5'],
        '2xl': ['1.5rem', '1.5'],
        '3xl': ['1.875rem', '1.5'],
        '4xl': ['2.25rem', '1.5'],
        '5xl': ['3rem', '1.5'],
      },
      // Custom Shadows
      boxShadow: {
        'custom': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      },
      // Custom Transitions
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
      },
    },
  }
};


```

