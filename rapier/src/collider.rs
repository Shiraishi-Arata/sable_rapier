use crate::ActiveLevelColliderInfo;
use crate::scene::{LevelColliderID, SimulationSceneData};
use rapier3d::dynamics::MassProperties;
use rapier3d::geometry::{Shape, ShapeType, SharedShape, TypedShape};
use rapier3d::glamx::{DVec3, IVec3};
use rapier3d::parry::bounding_volume::{Aabb, BoundingSphere};
use rapier3d::prelude::*;
use std::f32::consts::PI;

const WORLD_SIZE: Real = 30_000_000.0;

#[derive(Debug, Clone, Copy)]
pub struct LevelCollider {
    pub id: Option<LevelColliderID>,
    pub is_static: bool,
    pub cached_aabb: Option<Aabb>,
}

impl LevelCollider {
    #[must_use]
    pub fn new(id: Option<LevelColliderID>, is_static: bool) -> Self {
        Self {
            id,
            is_static,
            cached_aabb: None,
        }
    }

    fn scaled(self, _scale: &Vector) -> Self {
        Self { ..self }
    }
}

pub fn compute_local_aabb_from_bounds(min: IVec3, max: IVec3, com: DVec3) -> Aabb {
    Aabb::new(
        (min.as_dvec3() - com).as_vec3(),
        (max.as_dvec3() + 1.0 - com).as_vec3(),
    )
}

pub fn update_collider_aabb(sim: &mut SimulationSceneData, info: &ActiveLevelColliderInfo) {
    let (Some(min), Some(max), Some(com)) = (
        info.local_bounds_min,
        info.local_bounds_max,
        info.center_of_mass,
    ) else {
        return;
    };

    let new_aabb = compute_local_aabb_from_bounds(min, max, com);
    let collider = sim.collider_set.get_mut(info.collider).unwrap();
    let existing = *collider.shape().as_shape::<LevelCollider>().unwrap();
    collider.set_shape(SharedShape::new(LevelCollider {
        cached_aabb: Some(new_aabb),
        ..existing
    }));
    // These colliders are created disabled (no voxel geometry yet). Now that
    // bounds and COM exist, allow it to take part in collision detection.
    collider.set_enabled(true);
}

/// Publish a new center of mass for an active collider and keep its cached
/// broad-phase AABB in sync.
///
/// Both `setLocalBounds` and this function are the only ways the pieces of
/// data used by `update_collider_aabb` become available. Refreshing here is
/// essential: a kinematic contraption's COM is published by
/// `setKinematicContraptionTransform`, and without this call a collider whose
/// bounds arrived first would keep `cached_aabb == None` forever and panic
/// Rapier's broad phase on the next step.
pub(crate) fn publish_center_of_mass(
    sim: &mut SimulationSceneData,
    info: &mut ActiveLevelColliderInfo,
    com: DVec3,
) {
    let aabb_missing = sim
        .collider_set
        .get(info.collider)
        .and_then(|collider| collider.shape().as_shape::<LevelCollider>())
        .is_none_or(|level| level.cached_aabb.is_none());

    let changed = info.center_of_mass != Some(com);
    info.center_of_mass = Some(com);

    if changed || aabb_missing {
        update_collider_aabb(sim, info);
    }
}

impl RayCast for LevelCollider {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &rapier3d::parry::query::Ray,
        _max_time_of_impact: Real,
        _solid: bool,
    ) -> Option<rapier3d::parry::query::RayIntersection> {
        todo!()
    }
}

impl PointQuery for LevelCollider {
    fn project_local_point(
        &self,
        _pt: Vector,
        _solid: bool,
    ) -> rapier3d::parry::query::PointProjection {
        todo!()
    }

    fn project_local_point_and_get_feature(
        &self,
        _pt: Vector,
    ) -> (rapier3d::parry::query::PointProjection, FeatureId) {
        todo!()
    }
}

impl Shape for LevelCollider {
    fn compute_local_aabb(&self) -> Aabb {
        if self.is_static {
            Aabb::new(
                Vec3::new(-WORLD_SIZE, -WORLD_SIZE, -WORLD_SIZE),
                Vec3::new(WORLD_SIZE, WORLD_SIZE, WORLD_SIZE),
            )
        } else {
            // A dynamic collider has no voxel geometry until both bounds and
            // center of mass have been published (see `update_collider_aabb`).
            // Until then report an empty AABB instead of panicking: broad phase
            // has nothing to pair, and narrow phase never sees bounds/COM that
            // were not set yet.
            self.cached_aabb
                .unwrap_or_else(|| Aabb::new(Vec3::ZERO, Vec3::ZERO))
        }
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        if self.is_static {
            BoundingSphere::new(Vec3::ZERO, WORLD_SIZE)
        } else {
            BoundingSphere::new(Vec3::ZERO, 1.0)
        }
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(*self)
    }

    fn scale_dyn(&self, scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        Some(Box::new(self.scaled(&scale)))
    }

    fn mass_properties(&self, _density: Real) -> MassProperties {
        MassProperties {
            inv_mass: 0.0,
            inv_principal_inertia: AngVector::new(0.0, 0.0, 0.0),
            local_com: Vec3::ZERO,
            principal_inertia_local_frame: Default::default(),
        }
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        0.25
    }

    fn ccd_angular_thickness(&self) -> Real {
        PI / 8.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::SableDispatcher;
    use crate::event_handler::SableEventHandler;
    use crate::get_physics_state;
    use crate::hooks::SablePhysicsHooks;
    use crate::joints::SableJointSet;
    use crate::rope::RopeMap;
    use crate::scene::{
        ReportedCollisionBuffer, SableManifoldInfoMap, SableSceneData, SimulationSceneData,
    };
    use crate::{ActiveLevelColliderInfo, PHYSICS_STATE, PhysicsState};
    use rapier3d::dynamics::{
        CCDSolver, ImpulseJointSet, IslandManager, MultibodyJointSet, RigidBodyBuilder,
        RigidBodySet,
    };
    use rapier3d::geometry::{ColliderSet, DefaultBroadPhase, NarrowPhase};
    use rapier3d::parry::query::{DefaultQueryDispatcher, QueryDispatcher};
    use rapier3d::pipeline::PhysicsPipeline;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    fn init_physics_state() {
        PHYSICS_STATE.get_or_init(|| {
            RwLock::new(PhysicsState {
                integration_parameters: rapier3d::prelude::IntegrationParameters::default(),
                voxel_collider_map: crate::voxel_collider::VoxelColliderMap::new(),
            })
        });
    }

    fn headless_sim() -> (SimulationSceneData, Arc<RwLock<SableSceneData>>) {
        let sable_data = Arc::new(RwLock::new(SableSceneData {
            main_level_chunks: HashMap::new(),
            octree_chunks: HashMap::new(),
            joint_set: SableJointSet::new(),
            rope_map: RopeMap::default(),
            level_colliders: HashMap::new(),
            rigid_bodies: HashMap::new(),
        }));
        let manifold_info_map = Arc::new(SableManifoldInfoMap::default());
        let reported_collisions = Arc::new(ReportedCollisionBuffer::new());
        let dispatcher = SableDispatcher {
            sable_data: Arc::clone(&sable_data),
            manifold_info_map: Arc::clone(&manifold_info_map),
        };

        let sim_data = SimulationSceneData {
            pipeline: PhysicsPipeline::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::with_query_dispatcher(
                dispatcher.chain(DefaultQueryDispatcher),
            ),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            physics_hooks: SablePhysicsHooks {
                sable_data: Arc::clone(&sable_data),
                manifold_info_map,
                current_step_vm: None,
            },
            event_handler: SableEventHandler {
                reported_collisions,
            },
        };
        (sim_data, sable_data)
    }

    fn step(sim: &mut SimulationSceneData) {
        let params = get_physics_state();
        sim.pipeline.step(
            Vec3::ZERO,
            &params.integration_parameters,
            &mut sim.island_manager,
            &mut sim.broad_phase,
            &mut sim.narrow_phase,
            &mut sim.rigid_body_set,
            &mut sim.collider_set,
            &mut sim.impulse_joint_set,
            &mut sim.multibody_joint_set,
            &mut sim.ccd_solver,
            &sim.physics_hooks,
            &sim.event_handler,
        );
    }

    /// Regression: a freshly created dynamic LevelCollider has `cached_aabb == None`.
    /// Rapier's broad phase calls `compute_local_aabb` on it during `step`, which used
    /// to panic with "compute_local_aabb called before bounds were set".
    #[test]
    fn uninitialised_dynamic_level_collider_survives_step() {
        init_physics_state();
        let (mut sim, sable_data) = headless_sim();

        // Static ground uses the huge AABB, so it overlaps the dynamic collider.
        let ground_body = sim.rigid_body_set.insert(RigidBodyBuilder::fixed().build());
        let ground = ColliderBuilder::new(SharedShape::new(LevelCollider::new(None, true))).build();
        sim.collider_set
            .insert_with_parent(ground, ground_body, &mut sim.rigid_body_set);

        // A dynamic collider that was created but never had bounds/COM published.
        // Production creates it disabled so Rapier excludes it from collision
        // detection until `update_collider_aabb` enables it.
        let body = sim
            .rigid_body_set
            .insert(RigidBodyBuilder::dynamic().build());
        let col = ColliderBuilder::new(SharedShape::new(LevelCollider::new(Some(1), false)))
            .enabled(false)
            .build();
        let col_handle = sim
            .collider_set
            .insert_with_parent(col, body, &mut sim.rigid_body_set);
        sable_data
            .write()
            .unwrap()
            .level_colliders
            .insert(1, ActiveLevelColliderInfo::new(col_handle));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| step(&mut sim)));
        assert!(
            result.is_ok(),
            "step() panicked on an uninitialised dynamic collider"
        );
    }

    /// Bounds arriving before COM must still leave a usable collider: publishing
    /// COM through the kinematic-contraption path used to skip the AABB refresh,
    /// leaving `cached_aabb == None` forever.
    #[test]
    fn bounds_then_center_of_mass_publishes_aabb_and_enables_collider() {
        init_physics_state();
        let (mut sim, sable_data) = headless_sim();

        let body = sim
            .rigid_body_set
            .insert(RigidBodyBuilder::dynamic().build());
        let col = ColliderBuilder::new(SharedShape::new(LevelCollider::new(Some(2), false)))
            .enabled(false)
            .build();
        let col_handle = sim
            .collider_set
            .insert_with_parent(col, body, &mut sim.rigid_body_set);

        {
            let mut info = ActiveLevelColliderInfo::new(col_handle);
            // Bounds arrive first; COM is still None, so nothing can be cached yet.
            let empty_chunks: crate::scene::ChunkMap = HashMap::new();
            let physics_state = crate::get_physics_state();
            let collider_map = &physics_state.voxel_collider_map;
            info.set_local_bounds(
                IVec3::new(-2, -2, -2),
                IVec3::new(2, 2, 2),
                &empty_chunks,
                &collider_map,
            );
            update_collider_aabb(&mut sim, &info);
            assert!(!sim.collider_set.get(col_handle).unwrap().is_enabled());

            // Then COM is published, e.g. by setKinematicContraptionTransform.
            publish_center_of_mass(&mut sim, &mut info, DVec3::new(1.0, 2.0, 3.0));
            sable_data.write().unwrap().level_colliders.insert(2, info);
        }

        let collider = sim.collider_set.get(col_handle).unwrap();
        assert!(
            collider.is_enabled(),
            "collider should be enabled once ready"
        );
        let level = collider.shape().as_shape::<LevelCollider>().unwrap();
        assert!(
            level.cached_aabb.is_some(),
            "cached AABB should be published"
        );

        let expected = compute_local_aabb_from_bounds(
            IVec3::new(-2, -2, -2),
            IVec3::new(2, 2, 2),
            DVec3::new(1.0, 2.0, 3.0),
        );
        assert_eq!(level.cached_aabb.unwrap(), expected);
    }
}
