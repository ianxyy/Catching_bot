from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    PointCloud,
    RigidTransform,
    HydroelasticContactRepresentation,
    ContactResults,
    Rgba,
    RotationMatrix

)

class Log(LeafSystem):
    def __init__(self, original_plant, scene_graph, model_name, grasp_random_seed, velocity, roll, launching_position, launching_orientation, meshcat):
        LeafSystem.__init__(self)


        self.DeclareAbstractInputPort("contact_results_input", 
            AbstractValue.Make(ContactResults())                        #1 input
        )

        grasp = AbstractValue.Make({'gripper1': (RigidTransform(), 0), 'gripper2': (RigidTransform(), 0)}) # right:gripper1, left:gripper2
        self.DeclareAbstractInputPort("grasp_selection", grasp)                 #0 input_port

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.05, 0.7, self.log_results)

        self.plant = original_plant
        self.graph = scene_graph
        self.model_name = model_name
        self.grasp_random_seed = grasp_random_seed
        self.launching_position = launching_position
        self.launching_orientation = launching_orientation
        self.velocity = velocity
        self.roll = roll

        self.write = True

    def check_success(self, context):
        contact_results = self.get_input_port(0).Eval(context)
        
        # No contacts mean no grasp
        if contact_results.num_point_pair_contacts() == 0:
            print('Grasp Failed: did not touch')
            return False
        
        # Define names of gripper parts and the robot body
        gripper_1 = ['left_finger_1', 'right_finger_1', 'body_1']
        gripper_2 = ['left_finger_2', 'right_finger_2', 'body_2']
        obj_names = ['noodle','ring','cuboid']  # Add more names if necessary
        for i in range(300):
            obj_names.append(f'segment_{i}')
        bodyAB = []
        
        # Iterate through all contact pairs
        for i in range(contact_results.num_point_pair_contacts()):
            contact_info = contact_results.point_pair_contact_info(i)
            
            # Get the names of the bodies involved in contact
            bodyA_name = self.plant.GetBodyFromFrameId(self.plant.GetBodyFrameIdOrThrow(contact_info.bodyA_index())).name()
            bodyB_name = self.plant.GetBodyFromFrameId(self.plant.GetBodyFrameIdOrThrow(contact_info.bodyB_index())).name()
            AB = [bodyA_name, bodyB_name]
            
            # Check if contact involves only the gripper parts and not the robot body
            if bodyA_name not in gripper_1 + gripper_2 + obj_names or bodyB_name not in gripper_1 + gripper_2 + obj_names:
                print('Grasp Failed: collide with robot body')
                return False
            
            # Check if the contact is between the gripper parts themselves
            if ('left_finger_1' in AB or 'right_finger_1' in AB or 'left_finger_2' in AB or 'right_finger_2' in AB) and not (bodyA_name in obj_names or bodyB_name in obj_names):
                print('Grasp Failed: nothing between gripper fingers/fingers collide with other parts')
                return False
            
            bodyAB.append(bodyA_name)
            bodyAB.append(bodyB_name)

        gripper_1_contact_with_any_object = any(finger in bodyAB for finger in ['left_finger_1', 'right_finger_1']) and any(obj in bodyAB for obj in obj_names)
        gripper_2_contact_with_any_object = any(finger in bodyAB for finger in ['left_finger_2', 'right_finger_2']) and any(obj in bodyAB for obj in obj_names)
        if gripper_1_contact_with_any_object and gripper_2_contact_with_any_object:
            print("Grasp Succeed.")
            return True
        else:
            print("Grasp Failed: Both grippers are not effectively gripping the object.")
            return False
        
    def log_results(self, context, state):
        grasp = self.get_input_port(1).Eval(context)
        _, self.obj_catch_t = grasp['gripper1']
        if context.get_time() >= self.obj_catch_t + 0.02 and self.write:
            self.write = False
            result = 1 if self.check_success(context) else 0
            # with open('results_grasp_1_weightedangle_loss.txt', "a") as text_file:
            with open('results_ring_elliptical.txt', "a") as text_file:
                # Write the information to the file
                text_file.write(f"Object: {self.model_name}, Seed: {self.grasp_random_seed}, Result: {result}, vel: {self.velocity}, roll:{self.roll}, pos:{self.launching_position}, ori:{self.launching_orientation}\n")