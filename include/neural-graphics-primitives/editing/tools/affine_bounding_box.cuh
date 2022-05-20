#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/json_binding.h>

#include <json/json.hpp>

NGP_NAMESPACE_BEGIN

struct AffineBoundingBox {
	NGP_HOST_DEVICE AffineBoundingBox() {}

	NGP_HOST_DEVICE AffineBoundingBox(const Eigen::Vector3f& _center, const float size) {
        center = _center;
        scale = Eigen::Vector3f::Constant(size);
        u = scale.x() * u;
        v = scale.y() * v;
        w = scale.z() * w;
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
    }

	NGP_HOST_DEVICE AffineBoundingBox(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
        center = 0.5*(a+b);
        scale = b-a;
        u = scale.x() * u;
        v = scale.y() * v;
        w = scale.z() * w;
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
    }

    NGP_HOST_DEVICE Eigen::Vector3f diag() const {
		return rot_matrix*scale;
	}

    NGP_HOST_DEVICE void translate(Eigen::Vector3f translation) {
		center += translation;
	}

    NGP_HOST_DEVICE void set_center(Eigen::Vector3f _center) {
		center = _center;
	}

    NGP_HOST_DEVICE void scale_with_vector(Eigen::Vector3f scaling_vector) {
        scale = scale.cwiseProduct(scaling_vector);
        u = rot_matrix * scale.x() * Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        v = rot_matrix * scale.y() * Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        w = rot_matrix * scale.z() * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
    }

    NGP_HOST_DEVICE void rotate(Eigen::Matrix3f rotation) {
        rot_matrix = rotation * rot_matrix;
        u = rotation * u;
        v = rotation * v;
        w = rotation * w;
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
    }

    NGP_HOST_DEVICE void set_rotation(Eigen::Matrix3f rotation) {
        rot_matrix = rotation;
        u = rot_matrix * scale.x() * Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        v = rot_matrix * scale.y() * Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        w = rot_matrix * scale.z() * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
    }

    NGP_HOST_DEVICE void set_scale(Eigen::Vector3f _scale) {
        scale = _scale;
        u = rot_matrix * scale.x() * Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        v = rot_matrix * scale.y() * Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        w = rot_matrix * scale.z() * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
    }

	NGP_HOST_DEVICE bool contains(const Eigen::Vector3f& p) const{
		return
			u.dot(p-min) >= 0 && u.dot(p-min) < u.dot(u) &&
			v.dot(p-min) >= 0 && v.dot(p-min) < v.dot(v) &&
			w.dot(p-min) >= 0 && w.dot(p-min) < w.dot(w);
	}

    NGP_HOST_DEVICE void warp_box(const BoundingBox& aabb) {
        center = aabb.relative_pos(center);
        scale = scale.cwiseQuotient(aabb.diag());
        u = rot_matrix * scale.x() * Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        v = rot_matrix * scale.y() * Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        w = rot_matrix * scale.z() * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        min = -0.5*rot_matrix*scale+center;
        max = 0.5*rot_matrix*scale+center;
	}

    Eigen::Vector3f min;
    Eigen::Vector3f max;
    Eigen::Matrix3f rot_matrix = Eigen::Matrix3f::Identity();
    Eigen::Vector3f u = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f v = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f w = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f center;
    Eigen::Vector3f scale;

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["min"] = min;
        j["max"] = max;
        j["rot_matrix"] = rot_matrix;
        j["u"] = u;
        j["v"] = v;
        j["w"] = w;
        j["center"] = center;
        j["scale"] = scale;

        return j;
    }
};

inline void from_json(const nlohmann::json& j, AffineBoundingBox& x) {
	from_json(j.at("min"), x.min);
	from_json(j.at("max"), x.max);
    from_json(j.at("rot_matrix"), x.rot_matrix);
	from_json(j.at("u"), x.u);
    from_json(j.at("v"), x.v);
    from_json(j.at("w"), x.w);
    from_json(j.at("center"), x.center);
    from_json(j.at("scale"), x.scale);
}

inline void to_json(nlohmann::json& j, const AffineBoundingBox& x) {
	to_json(j["min"], x.min);
	to_json(j["max"], x.max);
	to_json(j["rot_matrix"], x.rot_matrix);
	to_json(j["u"], x.u);
    to_json(j["v"], x.v);
    to_json(j["w"], x.w);
    to_json(j["center"], x.center);
    to_json(j["scale"], x.scale );
}

NGP_NAMESPACE_END
