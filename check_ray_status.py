#!/usr/bin/env python3
"""
检查Ray集群状态的脚本
"""
import ray
import time

def check_ray_status():
    try:
        # 连接到Ray集群
        ray.init(address='auto', ignore_reinit_error=True)
        
        print("=== Ray集群状态 ===")
        print(f"Ray版本: {ray.__version__}")
        print(f"集群资源: {ray.cluster_resources()}")
        print(f"可用资源: {ray.available_resources()}")
        
        # 检查活跃任务
        print("\n=== 活跃任务 ===")
        tasks = ray.tasks()
        if tasks:
            for task in tasks:
                print(f"任务ID: {task['task_id']}")
                print(f"  状态: {task['state']}")
                print(f"  名称: {task['name']}")
                print(f"  开始时间: {task['start_time']}")
                print(f"  持续时间: {time.time() - task['start_time']:.2f}秒")
                print()
        else:
            print("没有活跃任务")
            
        # 检查活跃actors
        print("\n=== 活跃Actors ===")
        actors = ray.actors()
        if actors:
            for actor_id, actor in actors.items():
                print(f"Actor ID: {actor_id}")
                print(f"  状态: {actor['state']}")
                print(f"  类名: {actor['class_name']}")
                print(f"  节点ID: {actor['node_id']}")
                print()
        else:
            print("没有活跃actors")
            
    except Exception as e:
        print(f"检查Ray状态时出错: {e}")

if __name__ == "__main__":
    check_ray_status()


