#pragma once
#include <vector>
#include <builtin_types.h>
#include <windows.h>
#include <GL/gl.h>
#include "cuda.cuh"
#include "tiny_obj_loader.h"



class GameObject
{
public:
	GameObject();
	~GameObject();

	Attribute attribute;
	bool contact;

	std::vector<Face> faces;


	Vertex centerPosition;
};

