#seed 
upstream:

---

**links**: https://docs.python.org/3/library/logging.html

---

Brain Dump: 

- logging levels:  DEBUG, INFO, WARNING, ERROR, CRITICAL
- what does `logger = logging.getLogger(__name__)` do? https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
- changing format from config 

---
pip install not needed 

```python
# Configure logging
import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# ... Example usage 
logger.debug(f"Generated GPT prompt: {prompt.request}")
```

--- 




To maintain the DRY (Don't Repeat Yourself) principle and ensure maintainability in your logging setup across multiple files, you can create a centralized logging configuration module. This module will contain your logging setup, and you can import it into each of your files of interest. Here's how you can do it:

### Step 1: Create a Centralized Logging Module

1. **Create a New Python File for Logging Configuration**: Name it something like `log_config.py`.

2. **Add Your Logging Setup to This File**:
   ```python
   import os
   import logging
   from pythonjsonlogger import jsonlogger

   def setup_logging():
       log_dir = 'logs'
       if not os.path.exists(log_dir):
           os.makedirs(log_dir)

       log_file = os.path.join(log_dir, f"{__name__}.log")

       logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s %(levelname)s %(name)s %(message)s',
                           handlers=[logging.FileHandler(log_file),
                                     logging.StreamHandler()])

       class CustomJsonFormatter(jsonlogger.JsonFormatter):
           def add_fields(self, log_record, record, message_dict):
               super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
               log_record['level'] = record.levelname
               log_record['logger'] = record.name

       for handler in logging.getLogger().handlers:
           handler.setFormatter(CustomJsonFormatter())
   ```

### Step 2: Import and Use in Other Files

In each of your Python files where you want to implement logging:

1. **Import the `setup_logging` Function**:
   ```python
   from log_config import setup_logging
   ```

2. **Call the `setup_logging` Function at the Beginning**:
   ```python
# logging setup
setup_logging(__name__)
   ```

   This needs to be done before any logging calls or other imports that might trigger logging.

### Best Practices

- **Modularity**: Keep the logging configuration in its own module to avoid cluttering your application logic code.
- **Initialization**: Ensure `setup_logging()` is called only once at the start of your application. If you're using a web framework like Flask or Django, this should be part of the application/server initialization code.
- **Error Handling**: Consider adding error handling within the logging setup to manage any issues that arise during the configuration (e.g., issues creating directories, file permissions).
- **Flexibility**: You might want to pass parameters (like log level, log format) to the `setup_logging` function to allow different configurations based on the environment (development, production, etc.).

This approach will keep your logging configuration DRY, centralized, and easily maintainable. Each file will share the same logging setup, reducing redundancy and the risk of inconsistent logging configurations across your application.