import glfw
from OpenGL.GL import *
import sys


def initialize_glfw():
    if not glfw.init():
        print("failed to initialize glfw")
        sys.exit()
    print("Glfw initialized successfully")



def create_window(width=800, height=600, title="My First OpenGL Window"):
   
    #  window hints (optional)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    #  window creation
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        sys.exit()
    
    # window's context current
    glfw.make_context_current(window)
    print(f"Window created: {width}x{height}")
    return window


def key_callback(window, key, scancode, action, mods):
    """Handle keyboard input"""
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        print("ESC pressed - closing window")

def main():
  
    initialize_glfw()
    
    
    window = create_window()
    glfw.set_key_callback(window, key_callback)
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
    print(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode()}")
    
    #  main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
     
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        
        glfw.swap_buffers(window)
    
    
    glfw.terminate()

if __name__ == "__main__":
    main()