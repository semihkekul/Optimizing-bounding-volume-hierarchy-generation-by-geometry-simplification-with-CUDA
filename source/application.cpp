
#include <iostream>
#include <fstream>
#include <vector>
#include "cuda.cuh"
#include "device_launch_parameters.h"
#include "glut.h"
#include "camera.h"
#include "freeglut.h"
#include "tiny_obj_loader.h"
#include "gameObject.h"
#include "gpuWrapper.cuh"
#include <fstream>
#include <string>
bool g_RenderAabbs = true;
bool g_RenderTriangles = true;
uint primtiveIndexToDraw = 0;
stParameters gParameters;
void renderAABB(stAABB& aabb);
static stScene g_Scene;
GameObject* g_gameObject;
std::vector<stAABB> g_aabbs;
GameObject* g_remeshedGameObject;
void PrintInfo(const tinyobj::attrib_t& attrib,
	const std::vector<tinyobj::shape_t>& shapes,
	const std::vector<tinyobj::material_t>& materials) {
	std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
	std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
	std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2)
		<< std::endl;

	std::cout << "# of shapes    : " << shapes.size() << std::endl;
	std::cout << "# of materials : " << materials.size() << std::endl;

	for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
		printf("  v[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
			static_cast<const double>(attrib.vertices[3 * v + 0]),
			static_cast<const double>(attrib.vertices[3 * v + 1]),
			static_cast<const double>(attrib.vertices[3 * v + 2]));
	}

	for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
		printf("  n[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
			static_cast<const double>(attrib.normals[3 * v + 0]),
			static_cast<const double>(attrib.normals[3 * v + 1]),
			static_cast<const double>(attrib.normals[3 * v + 2]));
	}

	for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
		printf("  uv[%ld] = (%f, %f)\n", static_cast<long>(v),
			static_cast<const double>(attrib.texcoords[2 * v + 0]),
			static_cast<const double>(attrib.texcoords[2 * v + 1]));
	}

	// For each shape
	for (size_t i = 0; i < shapes.size(); i++) {
		printf("shape[%ld].name = %s\n", static_cast<long>(i),
			shapes[i].name.c_str());
		printf("Size of shape[%ld].indices: %lu\n", static_cast<long>(i),
			static_cast<unsigned long>(shapes[i].mesh.indices.size()));

		size_t index_offset = 0;

		assert(shapes[i].mesh.num_face_vertices.size() ==
			shapes[i].mesh.material_ids.size());

		printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
			static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

		// For each face
		for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
			size_t fnum = shapes[i].mesh.num_face_vertices[f];

			printf("  face[%ld].fnum = %ld\n", static_cast<long>(f),
				static_cast<unsigned long>(fnum));

			// For each vertex in the face
			for (size_t v = 0; v < fnum; v++) {
				tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
				printf("    face[%ld].v[%ld].idx = %d/%d/%d\n", static_cast<long>(f),
					static_cast<long>(v), idx.vertex_index, idx.normal_index,
					idx.texcoord_index);
			}

			printf("  face[%ld].material_id = %d\n", static_cast<long>(f),
				shapes[i].mesh.material_ids[f]);

			index_offset += fnum;
		}

		printf("shape[%ld].num_tags: %lu\n", static_cast<long>(i),
			static_cast<unsigned long>(shapes[i].mesh.tags.size()));
		for (size_t t = 0; t < shapes[i].mesh.tags.size(); t++) {
			printf("  tag[%ld] = %s ", static_cast<long>(t),
				shapes[i].mesh.tags[t].name.c_str());
			printf(" ints: [");
			for (size_t j = 0; j < shapes[i].mesh.tags[t].intValues.size(); ++j) {
				printf("%ld", static_cast<long>(shapes[i].mesh.tags[t].intValues[j]));
				if (j < (shapes[i].mesh.tags[t].intValues.size() - 1)) {
					printf(", ");
				}
			}
			printf("]");

			printf(" floats: [");
			for (size_t j = 0; j < shapes[i].mesh.tags[t].floatValues.size(); ++j) {
				printf("%f", static_cast<const double>(
					shapes[i].mesh.tags[t].floatValues[j]));
				if (j < (shapes[i].mesh.tags[t].floatValues.size() - 1)) {
					printf(", ");
				}
			}
			printf("]");

			printf(" strings: [");
			for (size_t j = 0; j < shapes[i].mesh.tags[t].stringValues.size(); ++j) {
				printf("%s", shapes[i].mesh.tags[t].stringValues[j].c_str());
				if (j < (shapes[i].mesh.tags[t].stringValues.size() - 1)) {
					printf(", ");
				}
			}
			printf("]");
			printf("\n");
		}
	}

	for (size_t i = 0; i < materials.size(); i++) {
		printf("material[%ld].name = %s\n", static_cast<long>(i),
			materials[i].name.c_str());
		printf("  material.Ka = (%f, %f ,%f)\n",
			static_cast<const double>(materials[i].ambient[0]),
			static_cast<const double>(materials[i].ambient[1]),
			static_cast<const double>(materials[i].ambient[2]));
		printf("  material.Kd = (%f, %f ,%f)\n",
			static_cast<const double>(materials[i].diffuse[0]),
			static_cast<const double>(materials[i].diffuse[1]),
			static_cast<const double>(materials[i].diffuse[2]));
		printf("  material.Ks = (%f, %f ,%f)\n",
			static_cast<const double>(materials[i].specular[0]),
			static_cast<const double>(materials[i].specular[1]),
			static_cast<const double>(materials[i].specular[2]));
		printf("  material.Tr = (%f, %f ,%f)\n",
			static_cast<const double>(materials[i].transmittance[0]),
			static_cast<const double>(materials[i].transmittance[1]),
			static_cast<const double>(materials[i].transmittance[2]));
		printf("  material.Ke = (%f, %f ,%f)\n",
			static_cast<const double>(materials[i].emission[0]),
			static_cast<const double>(materials[i].emission[1]),
			static_cast<const double>(materials[i].emission[2]));
		printf("  material.Ns = %f\n",
			static_cast<const double>(materials[i].shininess));
		printf("  material.Ni = %f\n", static_cast<const double>(materials[i].ior));
		printf("  material.dissolve = %f\n",
			static_cast<const double>(materials[i].dissolve));
		printf("  material.illum = %d\n", materials[i].illum);
		printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
		printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
		printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
		printf("  material.map_Ns = %s\n",
			materials[i].specular_highlight_texname.c_str());
		printf("  material.map_bump = %s\n", materials[i].bump_texname.c_str());
		printf("    bump_multiplier = %f\n", static_cast<const double>(materials[i].bump_texopt.bump_multiplier));
		printf("  material.map_d = %s\n", materials[i].alpha_texname.c_str());
		printf("  material.disp = %s\n", materials[i].displacement_texname.c_str());
		printf("  <<PBR>>\n");
		printf("  material.Pr     = %f\n", static_cast<const double>(materials[i].roughness));
		printf("  material.Pm     = %f\n", static_cast<const double>(materials[i].metallic));
		printf("  material.Ps     = %f\n", static_cast<const double>(materials[i].sheen));
		printf("  material.Pc     = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
		printf("  material.Pcr    = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
		printf("  material.aniso  = %f\n", static_cast<const double>(materials[i].anisotropy));
		printf("  material.anisor = %f\n", static_cast<const double>(materials[i].anisotropy_rotation));
		printf("  material.map_Ke = %s\n", materials[i].emissive_texname.c_str());
		printf("  material.map_Pr = %s\n", materials[i].roughness_texname.c_str());
		printf("  material.map_Pm = %s\n", materials[i].metallic_texname.c_str());
		printf("  material.map_Ps = %s\n", materials[i].sheen_texname.c_str());
		printf("  material.norm   = %s\n", materials[i].normal_texname.c_str());
		std::map<std::string, std::string>::const_iterator it(
			materials[i].unknown_parameter.begin());
		std::map<std::string, std::string>::const_iterator itEnd(
			materials[i].unknown_parameter.end());

		for (; it != itEnd; it++) {
			printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
		}
		printf("\n");
	}
}
GameObject* loadObjFile(const char* filename, Vertex center)
{
	FILE *fp;
	fp = fopen("original.c", "w");

	std::cout << "Loading " << filename << std::endl;

	tinyobj::attrib_t attrib;

	std::vector<tinyobj::material_t> materials;
	std::vector<tinyobj::shape_t> shapes;
	GameObject* g_gameObject = new GameObject;
	g_gameObject->centerPosition = center;

	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename, NULL, true);

	if (!err.empty()) { // `err` may contain warning message.
		std::cerr << err << std::endl;
	}


	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
		{
			int fv = shapes[s].mesh.num_face_vertices[f];
			Face face;
			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) 
			{
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				//tri.V[v].x = attrib.vertices[3 * idx.vertex_index + 0];
				//tri.V[v].y = attrib.vertices[3 * idx.vertex_index + 1];
				//tri.V[v].z = attrib.vertices[3 * idx.vertex_index + 2];
				//tri.N[v].x = attrib.normals[3 * idx.normal_index + 0];
				//tri.N[v].y = attrib.normals[3 * idx.normal_index + 1];
				//tri.N[v].z = attrib.normals[3 * idx.normal_index + 2];

				face.vertexIndices[v] = idx.vertex_index;
			}
			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];

			g_gameObject->faces.push_back(face);
		}

	}



	for (size_t v = 0; v < attrib.vertices.size() / 3; v++)
	{
		GLdouble vx = static_cast<const double>(attrib.vertices[3 * v + 0]);
		GLdouble vy = static_cast<const double>(attrib.vertices[3 * v + 1]);
		GLdouble vz = static_cast<const double>(attrib.vertices[3 * v + 2]);
		Vertex vertexPos = make_float3(vx, vy, vz);
		g_gameObject->attribute.vertices.push_back(vertexPos);
		fprintf(fp,"%f %f %f\n", vx, vy, vz);
		GLdouble nx = static_cast<const double>(attrib.normals[3 * v + 0]);
		GLdouble ny = static_cast<const double>(attrib.normals[3 * v + 1]);
		GLdouble nz = static_cast<const double>(attrib.normals[3 * v + 2]);
		Vertex normalVertex = make_float3(nx, ny, nz);
		g_gameObject->attribute.normals.push_back(normalVertex);

		float valX = vertexPos.x;
		float valY = vertexPos.y;
		float valZ = vertexPos.z;

		//i want to gather the whole AABB  of the stScene here, this is a slow process (maybe do it in GPU ???)
		//fisrt lower corner
		if (valX < g_Scene.min.x)
		{
			g_Scene.min.x = valX;
		}
		if (valY < g_Scene.min.y)
		{
			g_Scene.min.y = valY;
		}
		if (valZ < g_Scene.min.z)
		{
			g_Scene.min.z = valZ;
		}
		//now higher corner
		if (valX > g_Scene.max.x)
		{
			g_Scene.max.x = valX;
		}
		if (valY > g_Scene.max.y)
		{
			g_Scene.max.y = valY;
		}
		if (valZ > g_Scene.max.z)
		{
			g_Scene.max.z = valZ;
		}
	}
	fclose(fp);
	return g_gameObject;
}
void loadScene()
{
	PrintFunction;
	


	// TODO: load according to type written in objects.txt
	g_gameObject = loadObjFile(gParameters.objFileName.c_str(), make_float3(0, 3, 0));  // object mesh and AABB is created with this func.
	if (g_gameObject)
	{
		std::cout << "SCENE : " << g_Scene.min.x << " " << g_Scene.min.y << " " << g_Scene.min.z << " to " << g_Scene.max.x << " " << g_Scene.max.y << " " << g_Scene.max.z << std::endl;

		cudaMain(g_gameObject->attribute.vertices, g_Scene, g_gameObject->faces, gParameters);

		g_remeshedGameObject = new GameObject;
		primtiveIndexToDraw = getGameObjectTriangles(g_remeshedGameObject, g_aabbs);

	}
	else
	{
		printf("\nobject not loaded!\n");
	}
}

/************************************************************************************************************
 * Graphics related
 ***********************************************************************************************************/
Camera g_Camera;
class Window {
public:
	Window() {
		this->interval = 1000 / 60;		//60 FPS
		this->window_handle = -1;
	}
	int window_handle, interval;
	glm::ivec2 size;
	float window_aspect;
} window;


void reshape(int w, int h)
{
	if (h > 0) {
		window.size = glm::ivec2(w, h);
		window.window_aspect = float(w) / float(h);
	}
	g_Camera.SetViewport(0, 0, window.size.x, window.size.y);
}



//Keyboard input for camera, also handles exit case
void keyboardFunc(unsigned char c, int x, int y) {
	switch (c) {
	case 'w':
		g_Camera.Move(FORWARD);
		break;
	case 'a':
		g_Camera.Move(LEFT);
		break;
	case 's':
		g_Camera.Move(BACK);
		break;
	case 'd':
		g_Camera.Move(RIGHT);
		break;
	case 'q':
		g_Camera.Move(DOWN);
		break;
	case 'e':
		g_Camera.Move(UP);
		break;
	case '+':
		primtiveIndexToDraw+=20;

		break;
	case '-':
		primtiveIndexToDraw-=20;

		break;
	case 'x':
	case 27:
		exit(0);
		return;
	default:
		break;
	}
}
//Used when person clicks mouse
void CallBackMouseFunc(int button, int state, int x, int y) {
	g_Camera.SetPos(button, state, x, y);
}
//Used when person drags mouse around
void CallBackMotionFunc(int x, int y) {
	g_Camera.Move2D(x, y);
}

float gridSize = 10;

void drawFloorGrid()
{
	glColor3f(.3, .3, .3);
	glBegin(GL_QUADS);
	glVertex3f(-gridSize, -0.001, -gridSize);
	glVertex3f(-gridSize, -0.001, gridSize);
	glVertex3f(gridSize, -0.001, gridSize);
	glVertex3f(gridSize, -0.001, -gridSize);
	glEnd();

	glBegin(GL_LINES);
	for (int i = -gridSize; i <= gridSize; i++) {
		if (i == 0)
		{
			glColor3f(.6, .3, .3);
		}
		else
		{
			glColor3f(.25, .25, .25);
		};
		glVertex3f(i, 0, -gridSize);
		glVertex3f(i, 0, gridSize);

		if (i == 0)
		{
			glColor3f(.3, .3, .6);
		}
		else
		{
			glColor3f(.25, .25, .25);
		};
		glVertex3f(-gridSize, 0, i);
		glVertex3f(gridSize, 0, i);
	};
	glEnd();
}

#define RADPERDEG 0.0174533

void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D)
{
	double x = x2 - x1;
	double y = y2 - y1;
	double z = z2 - z1;
	double L = sqrt(x*x + y*y + z*z);

	GLUquadricObj *quadObj;

	glPushMatrix();

	glTranslated(x1, y1, z1);

	if ((x != 0.) || (y != 0.)) {
		glRotated(atan2(y, x) / RADPERDEG, 0., 0., 1.);
		glRotated(atan2(sqrt(x*x + y*y), z) / RADPERDEG, 0., 1., 0.);
	}
	else if (z<0) {
		glRotated(180, 1., 0., 0.);
	}

	glTranslatef(0, 0, L - 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, 2 * D, 0.0, 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, 2 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	glTranslatef(0, 0, -L + 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, D, D, L - 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, D, 32, 1);
	gluDeleteQuadric(quadObj);

	glPopMatrix();

}


void RenderString(float x, float y, void *font, const char* str)
{

	glRasterPos2f(x, y);

	glutBitmapString(font, (const unsigned char*)str);

}



void drawInfo()
{
	glPushMatrix();
	glColor3f(1, 1, 1);
	char buffer[256];
	sprintf(buffer, "idx%d, cam: %f %f %f",primtiveIndexToDraw, g_Camera.GetPosition().x, g_Camera.GetPosition().y,g_Camera.GetPosition().z);
	RenderString(-1.0f, .9f, GLUT_BITMAP_HELVETICA_18, buffer);

	memset(buffer, 0x0, 256);


	glPopMatrix();
}
void drawAxes(GLdouble length)
{
	glPushMatrix();
	glColor3f(1, 0, 0);
	glTranslatef(-length, 0, 0);
	Arrow(0, 0, 0, 2 * length, 0, 0, 0.1);
	glPopMatrix();

	glPushMatrix();
	glColor3f(0, 1, 0);
	glTranslatef(0, -length, 0);
	Arrow(0, 0, 0, 0, 2 * length, 0, 0.1);
	glPopMatrix();

	glPushMatrix();
	glColor3f(0, 0, 1);
	glTranslatef(0, 0, -length);
	Arrow(0, 0, 0, 0, 0, 2 * length, 0.1);
	glPopMatrix();
}
//#define DEBUG

void drawObjects()
{

	glPushMatrix();
	
	drawAxes(1);
	//glLineWidth(5.0f);
#ifdef DEBUG
	FILE *fp;
	fp = fopen("Output.txt", "w");
#endif
	glPushMatrix();
	glTranslatef(g_gameObject->centerPosition.x, g_gameObject->centerPosition.y, g_gameObject->centerPosition.z);
	glColor3f(0.0, 1.0, 0.0);
	for (auto face : g_gameObject->faces)
	{
		uint v0Idx = face.vertexIndices[0];
		uint v1Idx = face.vertexIndices[1];
		uint v2Idx = face.vertexIndices[2];

		const Vertex V0 = g_gameObject->attribute.vertices[v0Idx];
		const Vertex V1 = g_gameObject->attribute.vertices[v1Idx];
		const Vertex V2 = g_gameObject->attribute.vertices[v2Idx];

		glBegin(GL_TRIANGLES);
		glVertex3f(V0.x, V0.y, V0.z);
		glVertex3f(V1.x, V1.y, V1.z);
		glVertex3f(V2.x, V2.y, V2.z);

		glEnd();
#ifdef DEBUG
		fprintf(fp, "%f %f %f\n", V0.x, V0.y, V0.z);
		fprintf(fp, "%f %f %f\n", V1.x, V1.y, V1.z);
		fprintf(fp, "%f %f %f\n", V2.x, V2.y, V2.z);
#endif
	}
	glPopMatrix();


#ifdef DEBUG
	fprintf(fp, "===\n");
#endif

	glPushMatrix();
	glTranslatef(g_remeshedGameObject->centerPosition.x, g_remeshedGameObject->centerPosition.y, g_remeshedGameObject->centerPosition.z);
	
	glPointSize(3.0f);
	uint i = 0;
	for (auto face : g_remeshedGameObject->faces)
	{
		uint v0Idx = face.vertexIndices[0];
		uint v1Idx = face.vertexIndices[1];
		uint v2Idx = face.vertexIndices[2];

		const Vertex V0 = g_remeshedGameObject->attribute.vertices[v0Idx];
		const Vertex V1 = g_remeshedGameObject->attribute.vertices[v1Idx];
		const Vertex V2 = g_remeshedGameObject->attribute.vertices[v2Idx];

		glColor3f(1.0, 1.0, 0.0);
		glBegin(GL_POINTS);
		glVertex3f(V0.x, V0.y, V0.z);
		glVertex3f(V1.x, V1.y, V1.z);
		glVertex3f(V2.x, V2.y, V2.z);
		glEnd();

		if(g_RenderTriangles && (i < primtiveIndexToDraw || i > primtiveIndexToDraw + 19))
		{
			glColor3f(0.5, 0.5, 0.5);
			glBegin(GL_TRIANGLES);
			glVertex3f(V0.x, V0.y, V0.z);
			glVertex3f(V1.x, V1.y, V1.z);
			glVertex3f(V2.x, V2.y, V2.z);
			glEnd();
		}

#ifdef DEBUG
		fprintf(fp, "%f %f %f\n", V0.x, V0.y, V0.z);
		fprintf(fp, "%f %f %f\n", V1.x, V1.y, V1.z);
		fprintf(fp, "%f %f %f\n", V2.x, V2.y, V2.z);
#endif
		++i;
	}
	glPopMatrix();


	glPushMatrix();
	{
		glTranslatef(g_remeshedGameObject->centerPosition.x, g_remeshedGameObject->centerPosition.y, g_remeshedGameObject->centerPosition.z);
		glColor3f(1.0, 0.0, 1.0);
		
		for (int k = primtiveIndexToDraw; k < primtiveIndexToDraw + 20 && k < 1237; ++k)
		{
			auto face = g_remeshedGameObject->faces[k];
			uint v0Idx = face.vertexIndices[0];
			uint v1Idx = face.vertexIndices[1];
			uint v2Idx = face.vertexIndices[2];

			Vertex V0 = g_remeshedGameObject->attribute.vertices[v0Idx];
			Vertex V1 = g_remeshedGameObject->attribute.vertices[v1Idx];
			Vertex V2 = g_remeshedGameObject->attribute.vertices[v2Idx];

			glBegin(GL_TRIANGLES);
			glVertex3f(V0.x, V0.y, V0.z);
			glVertex3f(V1.x, V1.y, V1.z);
			glVertex3f(V2.x, V2.y, V2.z);
			glEnd();
			
		}
		
	}
	glPopMatrix();
		
#ifdef DEBUG
	fclose(fp);
	exit(23);
#endif
	glPopMatrix();
	
}
void renderAABB(stAABB& aabb)
{
	// top
	glBegin(GL_LINE_STRIP);
	glVertex3d(aabb.posMax.x, aabb.posMax.y, aabb.posMax.z);
	glVertex3d(aabb.posMax.x, aabb.posMax.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMax.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMax.y, aabb.posMax.z);
	glVertex3d(aabb.posMax.x, aabb.posMax.y, aabb.posMax.z);
	glEnd();

	// bottom
	glBegin(GL_LINE_STRIP);
	glVertex3d(aabb.posMin.x, aabb.posMin.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMin.y, aabb.posMax.z);
	glVertex3d(aabb.posMax.x, aabb.posMin.y, aabb.posMax.z);
	glVertex3d(aabb.posMax.x, aabb.posMin.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMin.y, aabb.posMin.z);
	glEnd();

	// side
	glBegin(GL_LINES);
	glVertex3d(aabb.posMax.x, aabb.posMax.y, aabb.posMax.z);
	glVertex3d(aabb.posMax.x, aabb.posMin.y, aabb.posMax.z);
	glVertex3d(aabb.posMax.x, aabb.posMax.y, aabb.posMin.z);
	glVertex3d(aabb.posMax.x, aabb.posMin.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMax.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMin.y, aabb.posMin.z);
	glVertex3d(aabb.posMin.x, aabb.posMax.y, aabb.posMax.z);
	glVertex3d(aabb.posMin.x, aabb.posMin.y, aabb.posMax.z);
	glEnd();
}
void display(void)
{

	glEnable(GL_CULL_FACE);
	glClearColor(0.6, 0.7, 0.9, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glEnable(GL_DEPTH_TEST);

	glViewport(0, 0, window.size.x, window.size.y);

	glm::mat4 model, view, projection;
	g_Camera.Update();
	g_Camera.GetMatricies(projection, view, model);

	glm::mat4 mvp = projection* view * model;	//Compute the mvp matrix

	drawInfo();
	glLoadMatrixf(glm::value_ptr(mvp));
	drawObjects();
	drawFloorGrid();
	if (g_RenderAabbs)
	{
		glColor3f(1, 0, 0);
		glTranslatef(g_remeshedGameObject->centerPosition.x, g_remeshedGameObject->centerPosition.y, g_remeshedGameObject->centerPosition.z);
		for (stAABB aabb : g_aabbs)
		{
			renderAABB(aabb);
		}
	}
	glutSwapBuffers();
}


void render_aabbs(int value)
{
	g_RenderAabbs = (bool)value;
}

void polygon_mode(int value)
{
	switch (value) {
	case 1:
		//glEnable(GL_DEPTH_TEST);
		//glEnable(GL_LIGHTING);
		//glDisable(GL_BLEND);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	case 2:
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glColor3f(1.0, 1.0, 1.0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		break;
	}
	glutPostRedisplay();
}

void triangles_mode(int value)
{
	g_RenderTriangles = (bool)value;
}

void main_menu(int value)
{
	if (value == 666)
		exit(0);
}

void initGraphics(int argc, char** argv)
{
	int submenu;
	int submenuAabb;
	int subMenuTriangles;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1200, 800);
	glutInitWindowPosition(20, 20);


	GLfloat light_ambient[] =
	{ 0.0, 0.0, 0.0, 1.0 };
	GLfloat light_diffuse[] =
	{ 1.0, 0.0, 0.0, 1.0 };
	GLfloat light_specular[] =
	{ 1.0, 1.0, 1.0, 1.0 };
	/* light_position is NOT default value */
	GLfloat light_position[] =
	{ 1.0, 1.0, 1.0, 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);

	glutCreateWindow("Semih Kekül - Thesis");
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(keyboardFunc);
	glutMouseFunc(CallBackMouseFunc);
	glutMotionFunc(CallBackMotionFunc);
	g_Camera.SetMode(FREE);
	g_Camera.SetPosition(glm::vec3(gParameters.eyePosition.x, gParameters.eyePosition.y, gParameters.eyePosition.z));
	g_Camera.SetLookAt(glm::vec3(0, 0, 0));
	g_Camera.SetClipping(.1, 1000);
	g_Camera.SetFOV(45);

	submenu = glutCreateMenu(polygon_mode);
	glutAddMenuEntry("Filled", 1);
	glutAddMenuEntry("Outline", 2);

	submenuAabb = glutCreateMenu(render_aabbs);
	glutAddMenuEntry("Render", 1);
	glutAddMenuEntry("Hide", 0);

	subMenuTriangles = glutCreateMenu(triangles_mode);
	glutAddMenuEntry("Render", 1);
	glutAddMenuEntry("Hide", 0);

	glutCreateMenu(main_menu);
	glutAddMenuEntry("Quit", 666);
	glutAddSubMenu("Polygon mode", submenu);
	glutAddSubMenu("AABB", submenuAabb);
	glutAddSubMenu("Triangles", subMenuTriangles);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void loadParameters()
{
	std::ifstream myfile("parameters.cfg");
	std::string line;

	if (myfile.is_open())
	{
		getline(myfile, gParameters.objFileName);
		
		getline(myfile, line);
		gParameters.theta.x = atof(line.c_str());
		
		getline(myfile, line);
		gParameters.theta.y= atof(line.c_str());

		getline(myfile, line);
		gParameters.eyePosition.x = atof(line.c_str());
		getline(myfile, line);
		gParameters.eyePosition.y = atof(line.c_str());
		getline(myfile, line);
		gParameters.eyePosition.z = atof(line.c_str());

		std::cout << "object file name is " << gParameters.objFileName << std::endl;
		std::cout << "theta from  " << gParameters.theta.y << " to "<< gParameters.theta.x <<std::endl;
		std::cout << "camera position : " << gParameters.eyePosition.x << " " << gParameters.eyePosition.y << " " << gParameters.eyePosition.z << std::endl;


	}
	myfile.close();
}

int main(int argc, char** argv)
{
	PrintFunction;

	loadParameters();

	initCUDA();
	g_Scene.max = make_float3(-100, -100, -100);
	g_Scene.min = make_float3(100, 100, 100);

	loadScene();

	initGraphics(argc, argv);

	glutMainLoop();

	getchar();
	return 0;
}