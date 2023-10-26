import pygame
import pymunk
import pymunk.pygame_util
import numpy as np

COLLTYPE_DEFAULT = 0
COLLTYPE_MOUSE = 1
COLLTYPE_BALL = 2

def get_body_type(static=False):
    body_type = pymunk.Body.DYNAMIC
    if static:
        body_type = pymunk.Body.STATIC
    return body_type


def create_rectangle(space,
        pos_x,pos_y,width,height,
        density=3,static=False):
    body = pymunk.Body(body_type=get_body_type(static))
    body.position = (pos_x,pos_y)
    shape = pymunk.Poly.create_box(body,(width,height))
    shape.density = density
    space.add(body,shape)
    return body, shape


def create_rectangle_bb(space, 
        left, bottom, right, top, 
        **kwargs):
    pos_x = (left + right) / 2
    pos_y = (top + bottom) / 2
    height = top - bottom
    width = right - left
    return create_rectangle(space, pos_x, pos_y, width, height, **kwargs)

def create_circle(space, pos_x, pos_y, radius, density=3, static=False):
    body = pymunk.Body(body_type=get_body_type(static))
    body.position = (pos_x, pos_y)
    shape = pymunk.Circle(body, radius=radius)
    shape.density = density
    shape.collision_type = COLLTYPE_BALL
    space.add(body, shape)
    return body, shape

def get_body_state(body):
    state = np.zeros(6, dtype=np.float32)
    state[:2] = body.position
    state[2] = body.angle
    state[3:5] = body.velocity
    state[5] = body.angular_velocity
    return state
