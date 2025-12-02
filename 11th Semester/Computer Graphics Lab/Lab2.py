import glfw
from OpenGL.GL import *
import numpy as np
import math


class Camera:
    def __init__(self, position=None, yaw=-90.0, pitch=0.0):
        self.position = np.array(position if position else [0.0, 0.0, 3.0], dtype=np.float32)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.yaw = yaw
        self.pitch = pitch
        
        self.movement_speed = 2.5
        self.mouse_sensitivity = 0.1
        
        self.update_camera_vectors()
    
    def get_view_matrix(self):
        """Returns the view matrix for the camera"""
        return self.look_at(self.position, self.position + self.front, self.up)
    
    def look_at(self, position, target, up):
        """Create a view matrix (look-at matrix)"""
        z = position - target
        z = z / np.linalg.norm(z)
        
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        
        y = np.cross(z, x)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x
        view[1, :3] = y
        view[2, :3] = z
        view[0, 3] = -np.dot(x, position)
        view[1, 3] = -np.dot(y, position)
        view[2, 3] = -np.dot(z, position)
        
        return view
    
    def process_keyboard(self, direction, delta_time):
        """Process keyboard input for camera movement"""
        velocity = self.movement_speed * delta_time
        
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "LEFT":
            self.position -= self.right * velocity
        if direction == "RIGHT":
            self.position += self.right * velocity
        if direction == "UP":
            self.position += self.world_up * velocity
        if direction == "DOWN":
            self.position -= self.world_up * velocity
    
    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """Process mouse movement for camera rotation"""
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0
        
        self.update_camera_vectors()
    
    def update_camera_vectors(self):
        """Update camera direction vectors based on yaw and pitch"""
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float32)
        
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)


def create_perspective_matrix(fov, aspect, near, far):
    """Create a perspective projection matrix"""
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    
    perspective = np.zeros((4, 4), dtype=np.float32)
    perspective[0, 0] = f / aspect
    perspective[1, 1] = f
    perspective[2, 2] = (far + near) / (near - far)
    perspective[2, 3] = (2.0 * far * near) / (near - far)
    perspective[3, 2] = -1.0
    
    return perspective


# Vertex shader with transformation matrices
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
"""

# Fragment shader
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec3 ourColor;

void main()
{
    FragColor = vec4(ourColor, 1.0);
}
"""


def compile_shader(source, shader_type):
    """Compile a shader"""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"Shader compilation error: {error}")
        return None
    
    return shader


def create_shader_program():
    """Create and link shader program"""
    vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        print(f"Program linking error: {error}")
        return None
    
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program


def create_cube_data():
    """Create vertex data for a colored cube"""
    vertices = np.array([
        # Position           # Color
        -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,  # Back face (red)
         0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
         0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
         0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
        -0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
        -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
        
        -0.5, -0.5,  0.5,   0.0, 1.0, 0.0,  # Front face (green)
         0.5, -0.5,  0.5,   0.0, 1.0, 0.0,
         0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
         0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
        -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
        -0.5, -0.5,  0.5,   0.0, 1.0, 0.0,
        
        -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,  # Left face (blue)
        -0.5,  0.5, -0.5,   0.0, 0.0, 1.0,
        -0.5, -0.5, -0.5,   0.0, 0.0, 1.0,
        -0.5, -0.5, -0.5,   0.0, 0.0, 1.0,
        -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
        
         0.5,  0.5,  0.5,   1.0, 1.0, 0.0,  # Right face (yellow)
         0.5,  0.5, -0.5,   1.0, 1.0, 0.0,
         0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
         0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
         0.5, -0.5,  0.5,   1.0, 1.0, 0.0,
         0.5,  0.5,  0.5,   1.0, 1.0, 0.0,
        
        -0.5, -0.5, -0.5,   1.0, 0.0, 1.0,  # Bottom face (magenta)
         0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
         0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
         0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
        -0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
        -0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
        
        -0.5,  0.5, -0.5,   0.0, 1.0, 1.0,  # Top face (cyan)
         0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
         0.5,  0.5,  0.5,   0.0, 1.0, 1.0,
         0.5,  0.5,  0.5,   0.0, 1.0, 1.0,
        -0.5,  0.5,  0.5,   0.0, 1.0, 1.0,
        -0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
    ], dtype=np.float32)
    
    return vertices


def _cuboid_vertices(center, size, color):
    """Helper: create 36 vertices (6 faces × 2 triangles × 3 verts) for an axis-aligned cuboid.

    center: (x,y,z), size: (sx, sy, sz), color: (r,g,b)
    returns: numpy array interleaved (pos.xyz + color.rgb)
    """
    cx, cy, cz = center
    sx, sy, sz = size
    # 8 corners
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    p = [
        (cx - hx, cy - hy, cz - hz),  # 0
        (cx + hx, cy - hy, cz - hz),  # 1
        (cx + hx, cy + hy, cz - hz),  # 2
        (cx - hx, cy + hy, cz - hz),  # 3
        (cx - hx, cy - hy, cz + hz),  # 4
        (cx + hx, cy - hy, cz + hz),  # 5
        (cx + hx, cy + hy, cz + hz),  # 6
        (cx - hx, cy + hy, cz + hz),  # 7
    ]

    c = color
    # faces (two triangles each): use indices into p
    faces = [
        (0, 1, 2, 3),  # back
        (5, 4, 7, 6),  # front
        (4, 0, 3, 7),  # left
        (1, 5, 6, 2),  # right
        (3, 2, 6, 7),  # top
        (4, 5, 1, 0),  # bottom
    ]

    verts = []
    for (a, b, cidx, d) in faces:
        # first triangle a, b, c
        verts.extend(list(p[a]) + list(color))
        verts.extend(list(p[b]) + list(color))
        verts.extend(list(p[cidx]) + list(color))
        # second triangle a, c, d
        verts.extend(list(p[a]) + list(color))
        verts.extend(list(p[cidx]) + list(color))
        verts.extend(list(p[d]) + list(color))

    return np.array(verts, dtype=np.float32)


def create_table_data(width=2.0, depth=1.0, top_thickness=0.1, height=1.0, leg_thickness=0.1):
    """Create a simple table: top (cuboid) plus 4 legs (cuboids).

    - width, depth: tabletop surface size
    - height: distance from floor (y = -1.0) to tabletop top surface
    - top_thickness: thickness of tabletop
    - leg_thickness: thickness (square) of each leg

    Coordinates:
    - floor is at y = -1.0 in this scene (same as the line grid)
    - table top top surface will be at y = -1.0 + height
    """
    verts = []

    # colors — wood-like
    top_color = [0.72, 0.50, 0.20]
    leg_color = [0.55, 0.33, 0.12]

    # Top surface center and size
    top_top_y = -1.0 + height
    top_center_y = top_top_y - top_thickness / 2.0

    top_center = (0.0, top_center_y, 0.0)
    top_size = (width, top_thickness, depth)
    verts.append(_cuboid_vertices(top_center, top_size, top_color))

    # Legs positions (relative to top corners, inset slightly)
    x_offset = width / 2.0 - leg_thickness / 2.0
    z_offset = depth / 2.0 - leg_thickness / 2.0
    # top_bottom = top_top_y - top_thickness
    leg_top = top_top_y - top_thickness / 2.0
    leg_bottom = -1.0  # floor
    leg_height = leg_top - leg_bottom

    leg_size = (leg_thickness, leg_height, leg_thickness)
    leg_y_center = leg_bottom + leg_height / 2.0

    leg_positions = [
        (-x_offset, leg_y_center, -z_offset),
        (x_offset, leg_y_center, -z_offset),
        (x_offset, leg_y_center, z_offset),
        (-x_offset, leg_y_center, z_offset),
    ]

    for pos in leg_positions:
        verts.append(_cuboid_vertices(pos, leg_size, leg_color))

    return np.concatenate(verts)


def create_floor_data():
    """Create a grid floor for reference"""
    vertices = []
    color = [0.3, 0.3, 0.3]
    
    # Create a grid
    grid_size = 20
    for i in range(-grid_size, grid_size + 1):
        # Lines parallel to X axis
        vertices.extend([i, -1.0, -grid_size] + color)
        vertices.extend([i, -1.0, grid_size] + color)
        
        # Lines parallel to Z axis
        vertices.extend([-grid_size, -1.0, i] + color)
        vertices.extend([grid_size, -1.0, i] + color)
    
    return np.array(vertices, dtype=np.float32)


class App:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.window = None
        self.camera = Camera(position=[0.0, 0.0, 5.0])
        
        # Mouse state
        self.first_mouse = True
        self.last_x = self.width / 2
        self.last_y = self.height / 2
        
        # Timing
        self.delta_time = 0.0
        self.last_frame = 0.0
        
        # Cube positions
        self.cube_positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 5.0, -15.0]),
            np.array([-1.5, -2.2, -2.5]),
            np.array([-3.8, -2.0, -12.3]),
            np.array([2.4, -0.4, -3.5]),
            np.array([-1.7, 3.0, -7.5]),
            np.array([1.3, -2.0, -2.5]),
            np.array([1.5, 2.0, -2.5]),
            np.array([1.5, 0.2, -1.5]),
            np.array([-1.3, 1.0, -1.5])
        ]
    
    def init_glfw(self):
        """Initialize GLFW and create window"""
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        
        self.window = glfw.create_window(self.width, self.height, "Camera Movement Demo", None, None)
        
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
        
        glfw.make_context_current(self.window)
        glfw.set_window_user_pointer(self.window, self)
        
        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        
        # Capture mouse
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        # Enable vsync
        glfw.swap_interval(1)
    
    @staticmethod
    def framebuffer_size_callback(window, width, height):
        glViewport(0, 0, width, height)
        app = glfw.get_window_user_pointer(window)
        app.width = width
        app.height = height
    
    @staticmethod
    def mouse_callback(window, xpos, ypos):
        app = glfw.get_window_user_pointer(window)
        
        if app.first_mouse:
            app.last_x = xpos
            app.last_y = ypos
            app.first_mouse = False
        
        xoffset = xpos - app.last_x
        yoffset = app.last_y - ypos  # Reversed since y-coordinates go from bottom to top
        
        app.last_x = xpos
        app.last_y = ypos
        
        app.camera.process_mouse_movement(xoffset, yoffset)
    
    def process_input(self):
        """Process keyboard input"""
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.process_keyboard("FORWARD", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.process_keyboard("BACKWARD", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.process_keyboard("LEFT", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.process_keyboard("RIGHT", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.camera.process_keyboard("UP", self.delta_time)
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.camera.process_keyboard("DOWN", self.delta_time)
    
    def create_model_matrix(self, position, angle=0.0):
        """Create a model matrix for positioning objects"""
        model = np.eye(4, dtype=np.float32)
        
        # Translation
        model[0, 3] = position[0]
        model[1, 3] = position[1]
        model[2, 3] = position[2]
        
        # Rotation
        if angle != 0.0:
            c = math.cos(angle)
            s = math.sin(angle)
            
            # Rotate around axis (1, 0.3, 0.5)
            axis = np.array([1.0, 0.3, 0.5])
            axis = axis / np.linalg.norm(axis)
            
            rotation = np.eye(4, dtype=np.float32)
            rotation[0, 0] = c + axis[0]**2 * (1 - c)
            rotation[0, 1] = axis[0] * axis[1] * (1 - c) - axis[2] * s
            rotation[0, 2] = axis[0] * axis[2] * (1 - c) + axis[1] * s
            rotation[1, 0] = axis[1] * axis[0] * (1 - c) + axis[2] * s
            rotation[1, 1] = c + axis[1]**2 * (1 - c)
            rotation[1, 2] = axis[1] * axis[2] * (1 - c) - axis[0] * s
            rotation[2, 0] = axis[2] * axis[0] * (1 - c) - axis[1] * s
            rotation[2, 1] = axis[2] * axis[1] * (1 - c) + axis[0] * s
            rotation[2, 2] = c + axis[2]**2 * (1 - c)
            
            model = np.dot(model, rotation)
        
        return model
    
    def run(self):
        """Main application loop"""
        self.init_glfw()
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Create shader program
        shader_program = create_shader_program()
        
        # Create table VAO (replace cubes with a table model)
        table_vertices = create_table_data(width=2.0, depth=1.0, top_thickness=0.1, height=1.0, leg_thickness=0.12)
        table_vao = glGenVertexArrays(1)
        table_vbo = glGenBuffers(1)

        glBindVertexArray(table_vao)
        glBindBuffer(GL_ARRAY_BUFFER, table_vbo)
        glBufferData(GL_ARRAY_BUFFER, table_vertices.nbytes, table_vertices, GL_STATIC_DRAW)

        # Position attribute (3 floats) + color attribute (3 floats) interleaved
        stride = 6 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        # Create floor VAO
        floor_vertices = create_floor_data()
        floor_vao = glGenVertexArrays(1)
        floor_vbo = glGenBuffers(1)
        
        glBindVertexArray(floor_vao)
        glBindBuffer(GL_ARRAY_BUFFER, floor_vbo)
        glBufferData(GL_ARRAY_BUFFER, floor_vertices.nbytes, floor_vertices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        print("Camera Controls:")
        print("  WASD - Move forward/left/backward/right")
        print("  Mouse - Look around")
        print("  Space - Move up")
        print("  Shift - Move down")
        print("  ESC - Exit")
        
        # Render loop
        while not glfw.window_should_close(self.window):
            # Calculate delta time
            current_frame = glfw.get_time()
            self.delta_time = current_frame - self.last_frame
            self.last_frame = current_frame
            
            # Input
            self.process_input()
            
            # Render
            glClearColor(0.1, 0.1, 0.15, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glUseProgram(shader_program)
            
            # Create transformation matrices
            view = self.camera.get_view_matrix()
            projection = create_perspective_matrix(45.0, self.width / self.height, 0.1, 100.0)
            
            # Set uniforms
            view_loc = glGetUniformLocation(shader_program, "view")
            proj_loc = glGetUniformLocation(shader_program, "projection")
            model_loc = glGetUniformLocation(shader_program, "model")
            
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.T)
            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.T)
            
            # Draw floor
            glBindVertexArray(floor_vao)
            model = np.eye(4, dtype=np.float32)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T)
            glDrawArrays(GL_LINES, 0, len(floor_vertices) // 6)
            
            # Draw table (single model)
            glBindVertexArray(table_vao)
            model = np.eye(4, dtype=np.float32)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.T)
            # number of vertices = len(table_vertices) // 6
            glDrawArrays(GL_TRIANGLES, 0, len(table_vertices) // 6)
            
            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        # Cleanup
        glDeleteVertexArrays(1, [table_vao])
        glDeleteBuffers(1, [table_vbo])
        glDeleteVertexArrays(1, [floor_vao])
        glDeleteBuffers(1, [floor_vbo])
        glDeleteProgram(shader_program)
        
        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()