#include <GL/freeglut.h>
#include <GL/glu.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{
	constexpr float Pi = 3.14159265358979323846f;
	constexpr int WindowWidth = 1280;
	constexpr int WindowHeight = 800;
	constexpr int TrackSamples = 4096;
	constexpr float TrackHalfWidth = 0.72f;

	struct Vec3
	{
		float x;
		float y;
		float z;

		Vec3 operator+(const Vec3& other) const { return { x + other.x, y + other.y, z + other.z }; }
		Vec3 operator-(const Vec3& other) const { return { x - other.x, y - other.y, z - other.z }; }
		Vec3 operator*(float scale) const { return { x * scale, y * scale, z * scale }; }
	};

	struct TrackPoint
	{
		Vec3 position;
		Vec3 tangent;
		Vec3 side;
		Vec3 up;
		float distance;
	};

	struct SupportBent
	{
		Vec3 leftFoot;
		Vec3 rightFoot;
		Vec3 leftTop;
		Vec3 rightTop;
		bool leftClear;
		bool rightClear;
	};

	std::vector<Vec3> controlPoints;
	std::vector<TrackPoint> track;
	std::vector<SupportBent> supportBents;
	float rideProgress = 0.0f;
	float rideSpeed = 0.16f;
	float targetRideSpeed = 0.16f;
	Vec3 cameraEye{};
	Vec3 cameraTarget{};
	Vec3 cameraUp{ 0.0f, 1.0f, 0.0f };
	bool rideCameraInitialized = false;
	float orbitYaw = 35.0f;
	float orbitPitch = 27.0f;
	float orbitDistance = 120.0f;
	float trackLength = 0.0f;
	int lastUpdateMilliseconds = 0;
	int cameraMode = 0;
	int windowWidth = WindowWidth;
	int windowHeight = WindowHeight;
	int previousMouseX = 0;
	int previousMouseY = 0;
	bool dragging = false;
	bool paused = false;
	bool lightingEnabled = true;
	bool showHud = true;
	GLUquadric* quadric = nullptr;

	Vec3 cross(const Vec3& a, const Vec3& b)
	{
		return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
	}

	float dot(const Vec3& a, const Vec3& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	Vec3 normalized(const Vec3& value)
	{
		const float length = std::sqrt(dot(value, value));
		if (length < 0.0001f)
		{
			return { 0.0f, 0.0f, 1.0f };
		}
		return value * (1.0f / length);
	}

	Vec3 catmullRom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t)
	{
		const auto knotStep = [](const Vec3& a, const Vec3& b)
		{
			const Vec3 delta = b - a;
			return std::max(std::sqrt(std::sqrt(dot(delta, delta))), 0.001f);
		};
		const float t0 = 0.0f;
		const float t1 = t0 + knotStep(p0, p1);
		const float t2 = t1 + knotStep(p1, p2);
		const float t3 = t2 + knotStep(p2, p3);
		const float u = t1 + (t2 - t1) * t;
		const Vec3 a1 = p0 * ((t1 - u) / (t1 - t0)) + p1 * ((u - t0) / (t1 - t0));
		const Vec3 a2 = p1 * ((t2 - u) / (t2 - t1)) + p2 * ((u - t1) / (t2 - t1));
		const Vec3 a3 = p2 * ((t3 - u) / (t3 - t2)) + p3 * ((u - t2) / (t3 - t2));
		const Vec3 b1 = a1 * ((t2 - u) / (t2 - t0)) + a2 * ((u - t0) / (t2 - t0));
		const Vec3 b2 = a2 * ((t3 - u) / (t3 - t1)) + a3 * ((u - t1) / (t3 - t1));
		return b1 * ((t2 - u) / (t2 - t1)) + b2 * ((u - t1) / (t2 - t1));
	}

	void buildCoasterLayout()
	{
		controlPoints = {
			{ -48.0f, 5.0f, -24.0f }, { -45.0f, 7.0f, -32.0f }, { -37.0f, 13.0f, -38.0f },
			{ -27.0f, 19.0f, -40.0f }, { -17.0f, 22.0f, -40.0f }, { -9.0f, 18.0f, -40.0f },
			{ -3.0f, 9.0f, -39.0f }, { 4.0f, 5.0f, -38.0f }, { 14.0f, 11.0f, -36.0f },
			{ 24.0f, 18.0f, -32.0f }, { 33.0f, 16.0f, -26.0f }, { 41.0f, 10.0f, -18.0f },
			{ 47.0f, 6.0f, -8.0f }, { 50.0f, 5.0f, 4.0f }, { 48.0f, 6.0f, 16.0f },
			{ 42.0f, 8.0f, 26.0f }, { 33.0f, 10.0f, 33.0f }, { 23.0f, 13.0f, 38.0f },
			{ 13.0f, 15.0f, 39.0f }, { 3.0f, 10.0f, 37.0f }, { -7.0f, 7.0f, 33.0f },
			{ -17.0f, 8.0f, 27.0f }, { -25.0f, 10.0f, 18.0f }, { -29.0f, 8.0f, 9.0f },
			{ -28.0f, 6.0f, 1.0f }, { -22.0f, 7.0f, -4.0f }, { -15.0f, 13.0f, -5.0f },
			{ -9.0f, 9.0f, -3.0f }, { -5.0f, 8.0f, -8.0f }, { -8.0f, 6.0f, -15.0f },
			{ -17.0f, 7.0f, -19.0f }, { -28.0f, 9.0f, -17.0f }, { -38.0f, 10.0f, -12.0f },
			{ -45.0f, 8.0f, -13.0f }, { -49.0f, 6.0f, -18.0f }
		};
	}

	Vec3 rotateAroundAxis(const Vec3& value, const Vec3& axis, float angle)
	{
		const float cosine = std::cos(angle);
		const float sine = std::sin(angle);
		return value * cosine + cross(axis, value) * sine + axis * (dot(axis, value) * (1.0f - cosine));
	}

	void buildTrack()
	{
		track.clear();
		trackLength = 0.0f;
		if (controlPoints.size() < 4) return;

		const int pointCount = static_cast<int>(controlPoints.size());
		track.reserve(TrackSamples);
		for (int sample = 0; sample < TrackSamples; ++sample)
		{
			const float scaled = static_cast<float>(sample) * pointCount / TrackSamples;
			const int segment = static_cast<int>(scaled);
			const float t = scaled - segment;
			const Vec3& p0 = controlPoints[(segment - 1 + pointCount) % pointCount];
			const Vec3& p1 = controlPoints[segment % pointCount];
			const Vec3& p2 = controlPoints[(segment + 1) % pointCount];
			const Vec3& p3 = controlPoints[(segment + 2) % pointCount];
			const Vec3 position = catmullRom(p0, p1, p2, p3, t);
			const Vec3 tangent = normalized(catmullRom(p0, p1, p2, p3, t + 0.001f) -
				catmullRom(p0, p1, p2, p3, t - 0.001f));
			track.push_back({ position, tangent, {}, {}, 0.0f });
		}

		for (std::size_t i = 1; i < track.size(); ++i)
		{
			const Vec3 delta = track[i].position - track[i - 1].position;
			trackLength += std::sqrt(dot(delta, delta));
			track[i].distance = trackLength;
		}
		const Vec3 closingSegment = track.front().position - track.back().position;
		trackLength += std::sqrt(dot(closingSegment, closingSegment));
		if (trackLength <= 0.0f) return;

		const Vec3 worldUp{ 0.0f, 1.0f, 0.0f };
		Vec3 initialSide = cross(worldUp, track.front().tangent);
		if (dot(initialSide, initialSide) < 0.001f)
		{
			initialSide = cross({ 1.0f, 0.0f, 0.0f }, track.front().tangent);
		}
		track.front().side = normalized(initialSide);
		track.front().up = normalized(cross(track.front().tangent, track.front().side));

		for (std::size_t i = 1; i < track.size(); ++i)
		{
			const Vec3 previousTangent = track[i - 1].tangent;
			const Vec3 tangent = track[i].tangent;
			const Vec3 rotation = cross(previousTangent, tangent);
			const float sine = std::sqrt(dot(rotation, rotation));
			const float cosine = std::clamp(dot(previousTangent, tangent), -1.0f, 1.0f);
			Vec3 side = track[i - 1].side;
			if (sine > 0.00001f)
			{
				side = rotateAroundAxis(side, rotation * (1.0f / sine), std::atan2(sine, cosine));
			}
			side = normalized(side - tangent * dot(side, tangent));
			if (dot(side, side) < 0.5f)
			{
				side = normalized(cross(worldUp, tangent));
			}
			track[i].side = side;
			track[i].up = normalized(cross(tangent, side));
		}

		const Vec3 firstSide = track.front().side;
		const Vec3 lastSide = track.back().side;
		const float seamAngle = std::atan2(dot(track.front().tangent, cross(lastSide, firstSide)),
			dot(lastSide, firstSide));
		for (TrackPoint& point : track)
		{
			const float seamCorrection = seamAngle * point.distance / trackLength;
			point.side = normalized(rotateAroundAxis(point.side, point.tangent, seamCorrection));
			point.up = normalized(cross(point.tangent, point.side));
		}

		for (std::size_t i = 0; i < track.size(); ++i)
		{
			const std::size_t previous = i == 0 ? track.size() - 1 : i - 1;
			const std::size_t next = (i + 1) % track.size();
			float span = track[next].distance - track[previous].distance;
			if (span <= 0.0f) span += trackLength;
			const Vec3 curvature = (track[next].tangent - track[previous].tangent) * (2.0f / std::max(span, 0.001f));
			const float lateralCurvature = dot(curvature, track[i].side);
			const float bank = std::clamp(std::atan2(lateralCurvature * 16.0f * 16.0f, 9.81f), -0.48f, 0.48f);
			track[i].side = normalized(rotateAroundAxis(track[i].side, track[i].tangent, bank));
			track[i].up = normalized(cross(track[i].tangent, track[i].side));
		}
	}

	TrackPoint trackAtDistance(float distance)
	{
		if (track.empty() || trackLength <= 0.0f) return {};
		distance = std::fmod(distance, trackLength);
		if (distance < 0.0f) distance += trackLength;
		const auto next = std::lower_bound(track.begin(), track.end(), distance,
			[](const TrackPoint& point, float target) { return point.distance < target; });
		const std::size_t nextIndex = next == track.end() ? 0 : static_cast<std::size_t>(next - track.begin());
		const std::size_t previousIndex = nextIndex == 0 ? track.size() - 1 : nextIndex - 1;
		const TrackPoint& a = track[previousIndex];
		const TrackPoint& b = track[nextIndex];
		float aDistance = a.distance;
		float bDistance = b.distance;
		float sampleDistance = distance;
		if (nextIndex == 0)
		{
			bDistance = trackLength;
			if (sampleDistance < aDistance) sampleDistance += trackLength;
		}
		const float blend = bDistance > aDistance ? std::clamp((sampleDistance - aDistance) / (bDistance - aDistance), 0.0f, 1.0f) : 0.0f;
		TrackPoint result;
		result.position = a.position + (b.position - a.position) * blend;
		result.tangent = normalized(a.tangent + (b.tangent - a.tangent) * blend);
		result.side = normalized(a.side + (b.side - a.side) * blend);
		result.up = normalized(cross(result.tangent, result.side));
		result.side = normalized(cross(result.up, result.tangent));
		result.distance = distance;
		return result;
	}

	TrackPoint trackAt(float progress)
	{
		progress -= std::floor(progress);
		return trackAtDistance(progress * trackLength);
	}

	void cameraGoal(Vec3& eye, Vec3& target, Vec3& up)
	{
		if (cameraMode == 0)
		{
			const float yaw = orbitYaw * Pi / 180.0f;
			const float pitch = orbitPitch * Pi / 180.0f;
			target = { 0.0f, 10.0f, 0.0f };
			eye = {
				target.x + orbitDistance * std::cos(pitch) * std::sin(yaw),
				target.y + orbitDistance * std::sin(pitch),
				target.z + orbitDistance * std::cos(pitch) * std::cos(yaw)
			};
			up = { 0.0f, 1.0f, 0.0f };
			return;
		}

		const TrackPoint cart = trackAt(rideProgress);
		up = cart.up;
		if (cameraMode == 1)
		{
			eye = cart.position - cart.tangent * 8.0f + cart.up * 3.4f;
			target = cart.position + cart.up * 0.6f;
		}
		else
		{
			eye = cart.position + cart.up * 1.15f;
			target = eye + cart.tangent * 9.0f + cart.up * 0.8f;
		}
	}

	void updateRideCamera(float deltaSeconds)
	{
		Vec3 desiredEye;
		Vec3 desiredTarget;
		Vec3 desiredUp;
		cameraGoal(desiredEye, desiredTarget, desiredUp);
		if (!rideCameraInitialized)
		{
			cameraEye = desiredEye;
			cameraTarget = desiredTarget;
			cameraUp = desiredUp;
			rideCameraInitialized = true;
			return;
		}
		const float blend = 1.0f - std::exp(-deltaSeconds / 0.12f);
		cameraEye = cameraEye + (desiredEye - cameraEye) * blend;
		cameraTarget = cameraTarget + (desiredTarget - cameraTarget) * blend;
		cameraUp = normalized(cameraUp + (desiredUp - cameraUp) * blend);
	}

	void setMaterial(float red, float green, float blue, float shine = 24.0f)
	{
		const GLfloat color[] = { red, green, blue, 1.0f };
		const GLfloat specular[] = { 0.78f, 0.82f, 0.88f, 1.0f };
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shine);
		glColor3f(red, green, blue);
	}

	float terrainHeight(float x, float z)
	{
		return 0.12f * std::sin(x * 0.13f) * std::cos(z * 0.11f) +
			0.07f * std::sin((x + z) * 0.19f);
	}

	void applyTrackFrame(const TrackPoint& point)
	{
		const GLfloat frame[] = {
			point.side.x, point.side.y, point.side.z, 0.0f,
			point.up.x, point.up.y, point.up.z, 0.0f,
			point.tangent.x, point.tangent.y, point.tangent.z, 0.0f,
			point.position.x, point.position.y, point.position.z, 1.0f
		};
		glMultMatrixf(frame);
	}

	void drawBeam(const Vec3& start, const Vec3& end, float radius)
	{
		const Vec3 direction = end - start;
		const float length = std::sqrt(dot(direction, direction));
		if (length < 0.001f || quadric == nullptr)
		{
			return;
		}
		const Vec3 axis = normalized(direction);
		const Vec3 rotationAxis = cross({ 0.0f, 0.0f, 1.0f }, axis);
		const float axisLength = std::sqrt(dot(rotationAxis, rotationAxis));
		const float angle = std::acos(std::clamp(axis.z, -1.0f, 1.0f)) * 180.0f / Pi;
		glPushMatrix();
		glTranslatef(start.x, start.y, start.z);
		if (axisLength > 0.001f)
		{
			glRotatef(angle, rotationAxis.x / axisLength, rotationAxis.y / axisLength, rotationAxis.z / axisLength);
		}
		else if (axis.z < 0.0f)
		{
			glRotatef(180.0f, 1.0f, 0.0f, 0.0f);
		}
		gluCylinder(quadric, radius, radius * 0.94, length, 12, 1);
		glPopMatrix();
	}

	void drawGround()
	{
		if (lightingEnabled) glEnable(GL_LIGHTING); else glDisable(GL_LIGHTING);
		glBegin(GL_QUADS);
		for (int x = -90; x < 90; x += 3)
		{
			for (int z = -90; z < 90; z += 3)
			{
				const float variation = 0.025f * std::sin(x * 0.31f + z * 0.17f);
				const GLfloat shade[] = { 0.17f + variation, 0.34f + variation, 0.19f + variation, 1.0f };
				glColor3f(shade[0], shade[1], shade[2]);
				for (const Vec3& vertex : { Vec3{ static_cast<float>(x), 0.0f, static_cast<float>(z) },
					Vec3{ static_cast<float>(x + 3), 0.0f, static_cast<float>(z) },
					Vec3{ static_cast<float>(x + 3), 0.0f, static_cast<float>(z + 3) },
					Vec3{ static_cast<float>(x), 0.0f, static_cast<float>(z + 3) } })
				{
					const float y = terrainHeight(vertex.x, vertex.z);
					const Vec3 normal = normalized({
						-(terrainHeight(vertex.x + 0.1f, vertex.z) - terrainHeight(vertex.x - 0.1f, vertex.z)),
						0.2f,
						-(terrainHeight(vertex.x, vertex.z + 0.1f) - terrainHeight(vertex.x, vertex.z - 0.1f)) });
					glNormal3f(normal.x, normal.y, normal.z);
					glVertex3f(vertex.x, y, vertex.z);
				}
			}
		}
		glEnd();

		glDisable(GL_LIGHTING);
		glLineWidth(1.4f);
		glBegin(GL_LINES);
		for (int patch = 0; patch < 440; ++patch)
		{
			const float x = -52.0f + static_cast<float>((patch * 37) % 1040) * 0.1f;
			const float z = -52.0f + static_cast<float>((patch * 71) % 1040) * 0.1f;
			const float y = terrainHeight(x, z) + 0.015f;
			const float height = 0.12f + static_cast<float>(patch % 5) * 0.035f;
			glColor3f(0.23f + (patch % 4) * 0.018f, 0.42f + (patch % 5) * 0.025f, 0.20f);
			glVertex3f(x, y, z);
			glVertex3f(x - 0.08f, y + height, z + 0.02f);
			glVertex3f(x, y, z);
			glVertex3f(x + 0.09f, y + height * 0.8f, z - 0.03f);
		}
		glEnd();
		glLineWidth(1.0f);
		if (lightingEnabled) glEnable(GL_LIGHTING);
	}

	bool supportBeamClear(const Vec3& start, const Vec3& end)
	{
		const Vec3 beam = end - start;
		const float beamLengthSquared = dot(beam, beam);
		if (beamLengthSquared < 0.001f) return false;
		for (const TrackPoint& railPoint : track)
		{
			const float along = std::clamp(dot(railPoint.position - start, beam) / beamLengthSquared, 0.0f, 0.82f);
			const Vec3 closest = start + beam * along;
			const Vec3 separation = railPoint.position - closest;
			if (dot(separation, separation) < 1.42f * 1.42f) return false;
		}
		return true;
	}

	void buildSupportLayout()
	{
		supportBents.clear();
		for (float distance = 0.0f; distance < trackLength; distance += 6.0f)
		{
			const TrackPoint point = trackAtDistance(distance);
			if (point.position.y < 5.2f || point.up.y < 0.28f) continue;
			const float horizontalTangent = std::sqrt(point.tangent.x * point.tangent.x + point.tangent.z * point.tangent.z);
			if (horizontalTangent < 0.42f) continue;

			const Vec3 lateral = normalized({ -point.tangent.z, 0.0f, point.tangent.x });
			if (dot(lateral, lateral) < 0.5f) continue;
			const Vec3 leftFootPosition = point.position - lateral * 2.35f;
			const Vec3 rightFootPosition = point.position + lateral * 2.35f;
			const Vec3 leftFoot{ leftFootPosition.x, terrainHeight(leftFootPosition.x, leftFootPosition.z), leftFootPosition.z };
			const Vec3 rightFoot{ rightFootPosition.x, terrainHeight(rightFootPosition.x, rightFootPosition.z), rightFootPosition.z };
			const Vec3 leftTop = point.position - lateral * 1.55f - Vec3{ 0.0f, 0.36f, 0.0f };
			const Vec3 rightTop = point.position + lateral * 1.55f - Vec3{ 0.0f, 0.36f, 0.0f };
			const bool leftClear = supportBeamClear(leftFoot, leftTop);
			const bool rightClear = supportBeamClear(rightFoot, rightTop);
			if (leftClear || rightClear)
			{
				supportBents.push_back({ leftFoot, rightFoot, leftTop, rightTop, leftClear, rightClear });
			}
		}
	}

	void drawSupports()
	{
		setMaterial(0.31f, 0.36f, 0.42f, 76.0f);
		for (const SupportBent& bent : supportBents)
		{
			for (const Vec3& foot : { bent.leftFoot, bent.rightFoot })
			{
				glPushMatrix();
				glTranslatef(foot.x, foot.y + 0.10f, foot.z);
				glScalef(0.70f, 0.20f, 0.70f);
				glutSolidCube(1.0);
				glPopMatrix();
			}
			if (bent.leftClear) drawBeam(bent.leftFoot, bent.leftTop, 0.17f);
			if (bent.rightClear) drawBeam(bent.rightFoot, bent.rightTop, 0.17f);
			if (bent.leftClear && bent.rightClear)
			{
				drawBeam(bent.leftTop, bent.rightTop, 0.14f);
				drawBeam(bent.leftFoot + (bent.leftTop - bent.leftFoot) * 0.48f,
					bent.rightFoot + (bent.rightTop - bent.rightFoot) * 0.48f, 0.09f);
				const Vec3 leftMid = bent.leftFoot + (bent.leftTop - bent.leftFoot) * 0.52f;
				const Vec3 rightMid = bent.rightFoot + (bent.rightTop - bent.rightFoot) * 0.52f;
				drawBeam(bent.leftFoot, rightMid, 0.075f);
				drawBeam(bent.rightFoot, leftMid, 0.075f);
			}

			setMaterial(0.66f, 0.70f, 0.75f, 96.0f);
			for (const Vec3& joint : { bent.leftTop, bent.rightTop })
			{
				glPushMatrix();
				glTranslatef(joint.x, joint.y, joint.z);
				glutSolidSphere(0.18, 10, 8);
				glPopMatrix();
			}
			setMaterial(0.31f, 0.36f, 0.42f, 76.0f);
		}
	}

	void drawRailTube(float lateralOffset, float heightOffset, float radius)
	{
		constexpr int RingSegments = 8;
		for (int ring = 0; ring < RingSegments; ++ring)
		{
			const float angleA = 2.0f * Pi * ring / RingSegments;
			const float angleB = 2.0f * Pi * (ring + 1) / RingSegments;
			glBegin(GL_QUAD_STRIP);
			for (std::size_t i = 0; i <= track.size(); ++i)
			{
				const TrackPoint& point = track[i % track.size()];
				for (float angle : { angleA, angleB })
				{
					const float sideWeight = std::cos(angle);
					const float upWeight = std::sin(angle);
					const Vec3 normal = point.side * sideWeight + point.up * upWeight;
					const Vec3 center = point.position + point.side * lateralOffset + point.up * heightOffset;
					glNormal3f(normal.x, normal.y, normal.z);
					glVertex3f(center.x + normal.x * radius, center.y + normal.y * radius, center.z + normal.z * radius);
				}
			}
			glEnd();
		}
	}

	void drawTrack()
	{
		if (track.empty()) return;
		setMaterial(0.68f, 0.72f, 0.77f, 105.0f);
		for (float side : { -TrackHalfWidth, TrackHalfWidth })
		{
			drawRailTube(side, 0.0f, 0.14f);
		}

		for (float distance = 0.0f; distance < trackLength; distance += 1.8f)
		{
			const TrackPoint point = trackAtDistance(distance);
			setMaterial(0.20f, 0.24f, 0.29f, 42.0f);
			glPushMatrix();
			glTranslatef(point.position.x, point.position.y, point.position.z);
			const GLfloat frame[] = {
				point.side.x, point.side.y, point.side.z, 0.0f,
				point.up.x, point.up.y, point.up.z, 0.0f,
				point.tangent.x, point.tangent.y, point.tangent.z, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f
			};
			glMultMatrixf(frame);
			glPushMatrix();
			glTranslatef(0.0f, -0.19f, 0.0f);
			glScalef(1.95f, 0.18f, 0.30f);
			glutSolidCube(1.0);
			glPopMatrix();

			setMaterial(0.68f, 0.72f, 0.77f, 105.0f);
			for (float lateral : { -TrackHalfWidth, TrackHalfWidth })
			{
				glPushMatrix();
				glTranslatef(lateral, -0.08f, 0.0f);
				glutSolidTorus(0.035, 0.17, 7, 12);
				glPopMatrix();
			}
			glPopMatrix();
		}
	}

	void drawSeat(float side)
	{
		setMaterial(0.78f, 0.055f, 0.045f, 42.0f);
		glPushMatrix();
		glTranslatef(side, 0.30f, -0.05f);
		glScalef(0.62f, 0.20f, 0.72f);
		glutSolidCube(1.0);
		glPopMatrix();
		glPushMatrix();
		glTranslatef(side, 0.70f, -0.40f);
		glScalef(0.58f, 0.72f, 0.18f);
		glutSolidCube(1.0);
		glPopMatrix();
		glPushMatrix();
		glTranslatef(side, 1.18f, -0.43f);
		glScalef(0.42f, 0.23f, 0.24f);
		glutSolidCube(1.0);
		glPopMatrix();

		setMaterial(0.12f, 0.14f, 0.17f, 56.0f);
		drawBeam({ side - 0.23f, 0.96f, 0.10f }, { side - 0.30f, 0.78f, 0.58f }, 0.055f);
		drawBeam({ side + 0.23f, 0.96f, 0.10f }, { side + 0.30f, 0.78f, 0.58f }, 0.055f);
		drawBeam({ side - 0.30f, 0.78f, 0.58f }, { side + 0.30f, 0.78f, 0.58f }, 0.065f);
		setMaterial(0.72f, 0.75f, 0.79f, 90.0f);
		glPushMatrix();
		glTranslatef(side, 0.78f, 0.58f);
		glutSolidSphere(0.09, 10, 8);
		glPopMatrix();
	}

	void drawCar(const TrackPoint& point)
	{
		glPushMatrix();
		applyTrackFrame(point);
		setMaterial(0.10f, 0.12f, 0.15f, 70.0f);
		glPushMatrix();
		glTranslatef(0.0f, -0.31f, 0.0f);
		glScalef(1.9f, 0.30f, 2.55f);
		glutSolidCube(1.0);
		glPopMatrix();
		setMaterial(0.72f, 0.045f, 0.035f, 88.0f);
		glPushMatrix();
		glTranslatef(0.0f, -0.08f, 0.0f);
		glScalef(1.82f, 0.28f, 2.48f);
		glutSolidCube(1.0);
		glPopMatrix();
		setMaterial(0.88f, 0.11f, 0.065f, 100.0f);
		for (float side : { -0.77f, 0.77f })
		{
			glPushMatrix();
			glTranslatef(side, 0.04f, 0.0f);
			glScalef(0.22f, 0.32f, 2.25f);
			glutSolidCube(1.0);
			glPopMatrix();
		}

		setMaterial(0.12f, 0.14f, 0.17f, 62.0f);
		for (float side : { -0.76f, 0.76f })
		{
			for (float along : { -0.88f, 0.88f })
			{
				glPushMatrix();
				glTranslatef(side, -0.18f, along);
				glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
				glutSolidTorus(0.075, 0.20, 10, 18);
				glPopMatrix();
				glPushMatrix();
				glTranslatef(side, -0.35f, along);
				glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
				glutSolidTorus(0.045, 0.15, 8, 14);
				glPopMatrix();
			}
		}
		setMaterial(0.72f, 0.75f, 0.79f, 95.0f);
		for (float along : { -0.88f, 0.88f })
		{
			drawBeam({ -0.86f, -0.14f, along }, { 0.86f, -0.14f, along }, 0.065f);
		}
		drawSeat(-0.43f);
		drawSeat(0.43f);
		glPopMatrix();
	}

	void drawTrainShadow(const TrackPoint& point)
	{
		glDisable(GL_LIGHTING);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDepthMask(GL_FALSE);
		glColor4f(0.035f, 0.045f, 0.055f, 0.28f);
		glBegin(GL_TRIANGLE_FAN);
		glVertex3f(point.position.x, terrainHeight(point.position.x, point.position.z) + 0.025f, point.position.z);
		for (int i = 0; i <= 32; ++i)
		{
			const float angle = 2.0f * Pi * i / 32.0f;
			glVertex3f(point.position.x + std::cos(angle) * 1.5f,
				terrainHeight(point.position.x, point.position.z) + 0.025f,
				point.position.z + std::sin(angle) * 2.5f);
		}
		glEnd();
		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
		if (lightingEnabled) glEnable(GL_LIGHTING);
	}

	void drawCart(const TrackPoint& point)
	{
		drawTrainShadow(point);
		const float spacing = trackLength > 0.0f ? 3.0f / trackLength : 0.03f;
		TrackPoint previous = point;
		for (int car = 0; car < 3; ++car)
		{
			const TrackPoint current = trackAt(rideProgress - car * spacing);
			if (car > 0)
			{
				setMaterial(0.68f, 0.72f, 0.77f, 90.0f);
				drawBeam(previous.position - previous.up * 0.20f, current.position - current.up * 0.20f, 0.075f);
			}
			drawCar(current);
			previous = current;
		}
	}

	void drawTree(float x, float z, float scale)
	{
		glPushMatrix();
		glTranslatef(x, terrainHeight(x, z), z);
		glScalef(scale, scale, scale);
		setMaterial(0.30f, 0.19f, 0.11f, 10.0f);
		glPushMatrix();
		glTranslatef(0.0f, 1.3f, 0.0f);
		glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
		gluCylinder(quadric, 0.27, 0.16, 2.7, 12, 2);
		glPopMatrix();
		setMaterial(0.15f, 0.34f, 0.13f, 16.0f);
		glPushMatrix();
		glTranslatef(-0.58f, 2.55f, 0.0f);
		glutSolidSphere(0.78, 16, 12);
		glTranslatef(1.12f, 0.18f, 0.05f);
		glutSolidSphere(0.88, 16, 12);
		glTranslatef(-0.52f, 0.64f, -0.04f);
		glutSolidSphere(1.08, 18, 14);
		glTranslatef(0.10f, 0.68f, 0.0f);
		glutSolidSphere(0.76, 16, 12);
		glPopMatrix();
		glPopMatrix();
	}

	void drawBush(float x, float z, float scale)
	{
		glPushMatrix();
		glTranslatef(x, terrainHeight(x, z) + 0.28f * scale, z);
		glScalef(scale, scale, scale);
		setMaterial(0.12f, 0.30f, 0.14f, 12.0f);
		glutSolidSphere(0.55, 12, 9);
		glTranslatef(0.62f, -0.04f, 0.10f);
		glutSolidSphere(0.44, 12, 9);
		glTranslatef(-1.15f, 0.02f, -0.16f);
		glutSolidSphere(0.42, 12, 9);
		glPopMatrix();
	}

	void drawDistantHills()
	{
		setMaterial(0.20f, 0.39f, 0.24f, 8.0f);
		for (int i = 0; i < 7; ++i)
		{
			const float x = -72.0f + i * 24.0f;
			glPushMatrix();
			glTranslatef(x, terrainHeight(x, -70.0f) + 2.5f, -72.0f);
			glScalef(15.0f, 8.0f + (i % 3) * 1.5f, 10.0f);
			glutSolidSphere(1.0, 20, 14);
			glPopMatrix();
			glPushMatrix();
			glTranslatef(x, terrainHeight(x, 70.0f) + 2.5f, 72.0f);
			glScalef(15.0f, 7.5f + ((i + 1) % 3) * 1.5f, 10.0f);
			glutSolidSphere(1.0, 20, 14);
			glPopMatrix();
		}
	}

	void drawWalkingPath()
	{
		setMaterial(0.48f, 0.46f, 0.39f, 12.0f);
		glBegin(GL_QUAD_STRIP);
		for (int i = 0; i <= 160; ++i)
		{
			const float angle = 2.0f * Pi * i / 160.0f;
			for (float radiusScale : { 1.0f, 1.065f })
			{
				const float x = 57.0f * radiusScale * std::cos(angle);
					const float z = 49.0f * radiusScale * std::sin(angle);
				glNormal3f(0.0f, 1.0f, 0.0f);
				glVertex3f(x, terrainHeight(x, z) + 0.035f, z);
			}
		}
		glEnd();
	}

	void drawEntrance()
	{
		const float z = -56.0f;
		setMaterial(0.18f, 0.20f, 0.23f, 45.0f);
		for (float x : { -4.2f, 4.2f })
		{
			glPushMatrix();
			glTranslatef(x, terrainHeight(x, z) + 2.4f, z);
			glScalef(0.75f, 4.8f, 0.85f);
			glutSolidCube(1.0);
			glPopMatrix();
		}
		setMaterial(0.78f, 0.09f, 0.045f, 76.0f);
		glPushMatrix();
		glTranslatef(0.0f, terrainHeight(0.0f, z) + 5.0f, z);
		glScalef(9.4f, 1.5f, 1.0f);
		glutSolidCube(1.0);
		glPopMatrix();
		setMaterial(0.96f, 0.66f, 0.12f, 94.0f);
		glPushMatrix();
		glTranslatef(0.0f, terrainHeight(0.0f, z) + 5.95f, z);
		glScalef(9.8f, 0.18f, 1.18f);
		glutSolidCube(1.0);
		glPopMatrix();

		glDisable(GL_LIGHTING);
		glColor3f(1.0f, 0.85f, 0.42f);
		glRasterPos3f(-2.65f, terrainHeight(0.0f, z) + 5.15f, z + 0.53f);
		const std::string sign = "THRILLRIDE PARK";
		for (unsigned char character : sign) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, character);
		if (lightingEnabled) glEnable(GL_LIGHTING);

		setMaterial(0.70f, 0.72f, 0.70f, 22.0f);
		glPushMatrix();
		glTranslatef(-8.2f, terrainHeight(-8.2f, z) + 1.1f, z - 1.6f);
		glScalef(3.2f, 2.2f, 2.8f);
		glutSolidCube(1.0);
		glPopMatrix();
		setMaterial(0.68f, 0.11f, 0.07f, 55.0f);
		glPushMatrix();
		glTranslatef(-8.2f, terrainHeight(-8.2f, z) + 2.25f, z - 1.6f);
		glScalef(3.5f, 0.18f, 3.0f);
		glutSolidCube(1.0);
		glPopMatrix();
	}

	void drawSmallBuilding(float x, float z, float scale, float red, float green, float blue)
	{
		const float ground = terrainHeight(x, z);
		setMaterial(0.69f, 0.70f, 0.66f, 18.0f);
		glPushMatrix();
		glTranslatef(x, ground + 1.1f * scale, z);
		glScalef(4.0f * scale, 2.2f * scale, 3.4f * scale);
		glutSolidCube(1.0);
		glPopMatrix();
		setMaterial(red, green, blue, 44.0f);
		glPushMatrix();
		glTranslatef(x, ground + 2.3f * scale, z);
		glScalef(4.4f * scale, 0.20f * scale, 3.8f * scale);
		glutSolidCube(1.0);
		glPopMatrix();
	}

	void drawFlowerBed(float x, float z, float scale)
	{
		const float ground = terrainHeight(x, z);
		setMaterial(0.25f, 0.28f, 0.16f, 10.0f);
		glPushMatrix();
		glTranslatef(x, ground + 0.08f * scale, z);
		glScalef(1.7f * scale, 0.12f * scale, 1.05f * scale);
		glutSolidSphere(1.0, 18, 10);
		glPopMatrix();
		const float flowerPositions[][2] = { {-0.9f, -0.3f}, {-0.45f, 0.35f}, {0.0f, -0.15f}, {0.48f, 0.36f}, {0.92f, -0.28f} };
		for (int i = 0; i < 5; ++i)
		{
			const float colors[][3] = { {0.95f, 0.72f, 0.20f}, {0.88f, 0.31f, 0.39f}, {0.96f, 0.88f, 0.67f} };
			setMaterial(colors[i % 3][0], colors[i % 3][1], colors[i % 3][2], 22.0f);
			glPushMatrix();
			glTranslatef(x + flowerPositions[i][0] * scale, ground + 0.28f * scale,
				z + flowerPositions[i][1] * scale);
			glutSolidSphere(0.15 * scale, 10, 8);
			glPopMatrix();
		}
	}

	void drawPark()
	{
		drawDistantHills();
		drawWalkingPath();
		drawFlowerBed(10.0f, 18.0f, 0.85f);
		drawFlowerBed(0.0f, 8.0f, 0.90f);
		drawFlowerBed(16.0f, -5.0f, 0.85f);
		for (int i = 0; i < 6; ++i)
		{
			const float x = -45.0f + i * 18.0f;
			if (std::abs(x) > 8.0f) drawTree(x, -53.0f, 0.82f + 0.10f * (i % 3));
			drawTree(x, 52.0f, 0.82f + 0.10f * ((i + 1) % 3));
		}
		for (int i = 0; i < 6; ++i)
		{
			const float z = -27.0f + i * 10.8f;
			drawTree(-62.0f, z, 0.86f + 0.10f * (i % 3));
			drawTree(62.0f, z, 0.86f + 0.10f * ((i + 1) % 3));
		}
		for (int i = 0; i < 11; ++i)
		{
			const float x = -40.0f + i * 8.0f;
			drawBush(x, -46.0f, 0.68f + 0.10f * (i % 3));
			drawBush(x, 45.0f, 0.68f + 0.10f * ((i + 1) % 3));
		}

		drawSmallBuilding(-21.0f, -49.0f, 1.0f, 0.78f, 0.32f, 0.10f);
		drawSmallBuilding(19.0f, -50.0f, 0.82f, 0.16f, 0.48f, 0.54f);
		drawSmallBuilding(57.0f, 31.0f, 0.90f, 0.70f, 0.19f, 0.12f);
		drawSmallBuilding(-57.0f, 32.0f, 0.78f, 0.24f, 0.40f, 0.58f);
		drawEntrance();
	}

	void drawString(float x, float y, const std::string& text, void* font = GLUT_BITMAP_HELVETICA_18)
	{
		glRasterPos2f(x, y);
		for (unsigned char character : text)
		{
			glutBitmapCharacter(font, character);
		}
	}

	void drawHudPanel(float x, float y, float width, float height, float red, float green, float blue, float alpha)
	{
		glColor4f(red, green, blue, alpha);
		glBegin(GL_QUADS);
		glVertex2f(x, y);
		glVertex2f(x + width, y);
		glVertex2f(x + width, y + height);
		glVertex2f(x, y + height);
		glEnd();
	}

	void drawHud()
	{
		if (!showHud) return;
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		gluOrtho2D(0.0, static_cast<double>(windowWidth), 0.0, static_cast<double>(windowHeight));
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		const float panelTop = static_cast<float>(windowHeight - 22);
		const float panelBottom = panelTop - 188.0f;
		drawHudPanel(22.0f, panelBottom, 370.0f, 188.0f, 0.025f, 0.045f, 0.075f, 0.89f);
		drawHudPanel(22.0f, panelTop - 4.0f, 370.0f, 4.0f, 0.98f, 0.48f, 0.08f, 1.0f);
		glColor3f(1.0f, 0.72f, 0.24f);
		drawString(42.0f, panelTop - 34.0f, "THRILLRIDE 3D", GLUT_BITMAP_HELVETICA_18);
		glColor3f(0.70f, 0.79f, 0.88f);
		drawString(42.0f, panelTop - 55.0f, "PARK OPERATIONS  /  RIDE CONTROL", GLUT_BITMAP_HELVETICA_12);

		glColor3f(paused ? 1.0f : 0.34f, paused ? 0.62f : 0.88f, paused ? 0.25f : 0.48f);
		drawString(42.0f, panelTop - 84.0f, paused ? "RIDE PAUSED" : "RIDE IN MOTION");
		std::ostringstream speedText;
		speedText << std::fixed << std::setprecision(0) << rideSpeed * 100.0f * 3.6f << " km/h";
		glColor3f(0.96f, 0.97f, 1.0f);
		drawString(230.0f, panelTop - 84.0f, speedText.str());

		const char* cameraName = cameraMode == 0 ? "ORBIT CAMERA" : (cameraMode == 1 ? "CHASE CAMERA" : "FRONT SEAT VIEW");
		glColor3f(0.68f, 0.78f, 0.88f);
		drawString(42.0f, panelTop - 111.0f, cameraName, GLUT_BITMAP_HELVETICA_12);
		drawString(230.0f, panelTop - 111.0f, lightingEnabled ? "LIGHTS  ON" : "LIGHTS  OFF", GLUT_BITMAP_HELVETICA_12);

		drawHudPanel(42.0f, panelBottom + 44.0f, 326.0f, 5.0f, 0.19f, 0.25f, 0.32f, 1.0f);
		drawHudPanel(42.0f, panelBottom + 44.0f, 326.0f * rideProgress, 5.0f, 1.0f, 0.49f, 0.08f, 1.0f);
		glColor3f(0.66f, 0.74f, 0.84f);
		drawString(42.0f, panelBottom + 20.0f, "SPACE pause   +/- speed   C camera   R reset", GLUT_BITMAP_HELVETICA_12);
		drawString(42.0f, 28.0f, "DRAG / ARROWS orbit   WHEEL zoom   L lighting   H interface   ESC exit", GLUT_BITMAP_HELVETICA_12);

		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		if (lightingEnabled) glEnable(GL_LIGHTING);
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
	}

	void drawSky()
	{
		glDisable(GL_LIGHTING);
		glDisable(GL_FOG);
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		constexpr float radius = 150.0f;
		for (int band = 0; band < 16; ++band)
		{
			const float lower = (Pi * 0.5f) * band / 16.0f;
			const float upper = (Pi * 0.5f) * (band + 1) / 16.0f;
			glBegin(GL_QUAD_STRIP);
			for (int segment = 0; segment <= 64; ++segment)
			{
				const float longitude = 2.0f * Pi * segment / 64.0f;
				for (float latitude : { lower, upper })
				{
					const float blend = latitude / (Pi * 0.5f);
					glColor3f(0.72f - 0.40f * blend, 0.84f - 0.31f * blend, 0.94f - 0.16f * blend);
					glVertex3f(radius * std::cos(latitude) * std::cos(longitude),
						radius * std::sin(latitude), radius * std::cos(latitude) * std::sin(longitude));
				}
			}
			glEnd();
		}
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		if (lightingEnabled) glEnable(GL_LIGHTING);
		glEnable(GL_FOG);
	}

	void applyCamera()
	{
		if (!rideCameraInitialized) updateRideCamera(0.0f);
		gluLookAt(cameraEye.x, cameraEye.y, cameraEye.z,
			cameraTarget.x, cameraTarget.y, cameraTarget.z,
			cameraUp.x, cameraUp.y, cameraUp.z);
	}

	void display()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		applyCamera();
		drawSky();

		const GLfloat lightPosition[] = { -0.45f, 0.84f, 0.58f, 0.0f };
		glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
		const GLfloat fillPosition[] = { 0.42f, 0.38f, -0.82f, 0.0f };
		glLightfv(GL_LIGHT1, GL_POSITION, fillPosition);
		drawGround();
		drawPark();
		drawSupports();
		drawTrack();
		drawCart(trackAt(rideProgress));
		drawHud();
		glutSwapBuffers();
	}

	void reshape(int width, int height)
	{
		windowWidth = std::max(1, width);
		windowHeight = std::max(1, height);
		glViewport(0, 0, windowWidth, windowHeight);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(55.0, static_cast<double>(windowWidth) / windowHeight, 0.1, 250.0);
		glMatrixMode(GL_MODELVIEW);
	}

	void update(int)
	{
		const int now = glutGet(GLUT_ELAPSED_TIME);
		const float deltaSeconds = lastUpdateMilliseconds == 0 ? 0.016f :
			std::clamp((now - lastUpdateMilliseconds) / 1000.0f, 0.0f, 0.10f);
		lastUpdateMilliseconds = now;
		const float speedBlend = 1.0f - std::exp(-deltaSeconds / 0.35f);
		rideSpeed += (targetRideSpeed - rideSpeed) * speedBlend;
		if (!paused && trackLength > 0.0f)
		{
			rideProgress += rideSpeed * 100.0f * deltaSeconds / trackLength;
			rideProgress -= std::floor(rideProgress);
		}
		updateRideCamera(deltaSeconds);
		glutPostRedisplay();
		glutTimerFunc(16, update, 0);
	}

	void keyboard(unsigned char key, int, int)
	{
		switch (key)
		{
		case 27:
			if (quadric != nullptr)
			{
				gluDeleteQuadric(quadric);
				quadric = nullptr;
			}
			std::exit(0);
			break;
		case ' ':
			paused = !paused;
			break;
		case '+':
		case '=':
			targetRideSpeed = std::min(0.55f, targetRideSpeed + 0.03f);
			break;
		case '-':
		case '_':
			targetRideSpeed = std::max(0.03f, targetRideSpeed - 0.03f);
			break;
		case 'c':
		case 'C':
			cameraMode = (cameraMode + 1) % 3;
			break;
		case 'l':
		case 'L':
			lightingEnabled = !lightingEnabled;
			if (lightingEnabled) glEnable(GL_LIGHTING); else glDisable(GL_LIGHTING);
			break;
		case 'h':
		case 'H':
			showHud = !showHud;
			break;
		case 'r':
		case 'R':
			rideProgress = 0.0f;
			paused = false;
			rideSpeed = 0.16f;
			targetRideSpeed = 0.16f;
			break;
		default:
			break;
		}
		glutPostRedisplay();
	}

	void specialKeys(int key, int, int)
	{
		if (key == GLUT_KEY_LEFT) orbitYaw -= 4.0f;
		if (key == GLUT_KEY_RIGHT) orbitYaw += 4.0f;
		if (key == GLUT_KEY_UP) orbitPitch = std::min(78.0f, orbitPitch + 3.0f);
		if (key == GLUT_KEY_DOWN) orbitPitch = std::max(5.0f, orbitPitch - 3.0f);
		glutPostRedisplay();
	}

	void mouseButton(int button, int state, int x, int y)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			dragging = state == GLUT_DOWN;
			previousMouseX = x;
			previousMouseY = y;
		}
		if (button == 3 && state == GLUT_DOWN) orbitDistance = std::max(20.0f, orbitDistance - 2.5f);
		if (button == 4 && state == GLUT_DOWN) orbitDistance = std::min(165.0f, orbitDistance + 2.5f);
		glutPostRedisplay();
	}

	void mouseMotion(int x, int y)
	{
		if (dragging)
		{
			orbitYaw += (x - previousMouseX) * 0.45f;
			orbitPitch = std::clamp(orbitPitch + (previousMouseY - y) * 0.35f, 5.0f, 78.0f);
			previousMouseX = x;
			previousMouseY = y;
			glutPostRedisplay();
		}
	}

	void initializeOpenGL()
	{
		glClearColor(0.52f, 0.74f, 0.91f, 1.0f);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glShadeModel(GL_SMOOTH);
		glEnable(GL_COLOR_MATERIAL);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHT1);
		const GLfloat ambient[] = { 0.24f, 0.27f, 0.31f, 1.0f };
		const GLfloat diffuse[] = { 0.98f, 0.91f, 0.79f, 1.0f };
		const GLfloat specular[] = { 0.95f, 0.96f, 1.0f, 1.0f };
		const GLfloat fillDiffuse[] = { 0.28f, 0.36f, 0.48f, 1.0f };
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
		glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
		glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, fillDiffuse);
		glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
		glEnable(GL_FOG);
		const GLfloat fogColor[] = { 0.54f, 0.71f, 0.84f, 1.0f };
		glFogi(GL_FOG_MODE, GL_EXP2);
		glFogfv(GL_FOG_COLOR, fogColor);
		glFogf(GL_FOG_DENSITY, 0.0016f);
		glHint(GL_FOG_HINT, GL_NICEST);
		glEnable(GL_NORMALIZE);
		quadric = gluNewQuadric();
		if (quadric != nullptr)
		{
			gluQuadricNormals(quadric, GLU_SMOOTH);
		}
	}
}

int main(int argc, char** argv)
{
	buildCoasterLayout();
	buildTrack();
	buildSupportLayout();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(WindowWidth, WindowHeight);
	glutInitWindowPosition(80, 60);
	glutCreateWindow("ThrillRide 3D | OpenGL Roller Coaster Simulation");
	initializeOpenGL();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeys);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMotion);
	glutTimerFunc(16, update, 0);
	glutMainLoop();

	if (quadric != nullptr)
	{
		gluDeleteQuadric(quadric);
	}
	return 0;
}
