"""
LIXIL AI Hub Global Deployment Manager

This module provides comprehensive global deployment capabilities for the LIXIL AI Hub Platform,
including multi-region deployment patterns, configuration management, infrastructure automation,
and compliance handling across different geographical regions.

Key Features:
- Multi-region deployment orchestration
- Regional compliance and data residency management
- Infrastructure as Code (IaC) automation
- Load balancing and traffic routing
- Monitoring and health checks across regions
- Disaster recovery and failover mechanisms

Author: LIXIL AI Hub Platform Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import os
import subprocess
from pathlib import Path

import aiohttp
import asyncpg
from pydantic import BaseModel, Field, validator
import boto3
import kubernetes
from kubernetes import client, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentRegion(str, Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"


class DeploymentEnvironment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class ComplianceRegime(str, Enum):
    """Data compliance regimes."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California
    PIPEDA = "pipeda"  # Canada
    PDPA = "pdpa"  # Singapore/Thailand
    LGPD = "lgpd"  # Brazil
    APPI = "appi"  # Japan


class InfrastructureProvider(str, Enum):
    """Infrastructure providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"


class ServiceType(str, Enum):
    """Types of services to deploy."""
    FRONTEND = "frontend"
    BACKEND_API = "backend_api"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    LOAD_BALANCER = "load_balancer"
    MONITORING = "monitoring"


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    region: DeploymentRegion
    compliance_regimes: List[ComplianceRegime]
    data_residency_required: bool
    primary_language: str
    supported_languages: List[str]
    timezone: str
    infrastructure_provider: InfrastructureProvider
    instance_types: Dict[ServiceType, str]
    scaling_limits: Dict[str, int]
    backup_retention_days: int
    monitoring_endpoints: List[str]


@dataclass
class DeploymentTarget:
    """Target configuration for deployment."""
    environment: DeploymentEnvironment
    region: DeploymentRegion
    version: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    rollback_version: Optional[str] = None


class GlobalDeploymentConfig(BaseModel):
    """Global deployment configuration."""
    project_name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+$")
    environments: List[DeploymentEnvironment]
    regions: List[DeploymentRegion]
    primary_region: DeploymentRegion
    disaster_recovery_region: DeploymentRegion
    load_balancer_strategy: str = Field(default="round_robin")
    auto_scaling_enabled: bool = Field(default=True)
    backup_enabled: bool = Field(default=True)
    monitoring_enabled: bool = Field(default=True)
    compliance_validation: bool = Field(default=True)

    @validator('regions')
    def validate_regions(cls, v, values):
        """Validate that primary and DR regions are in the regions list."""
        primary = values.get('primary_region')
        dr = values.get('disaster_recovery_region')
        
        if primary and primary not in v:
            raise ValueError("Primary region must be in regions list")
        if dr and dr not in v:
            raise ValueError("Disaster recovery region must be in regions list")
        
        return v


class ServiceDeploymentSpec(BaseModel):
    """Specification for deploying a service."""
    service_name: str = Field(..., min_length=1, max_length=100)
    service_type: ServiceType
    image: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    resource_requirements: Dict[str, str] = Field(default_factory=dict)
    health_check_path: str = Field(default="/health")
    replicas: int = Field(default=2, ge=1, le=100)
    auto_scaling: bool = Field(default=True)
    persistent_storage: bool = Field(default=False)
    storage_size: str = Field(default="10Gi")


class DeploymentStatus(BaseModel):
    """Status of a deployment."""
    deployment_id: str
    target: DeploymentTarget
    status: str  # "pending", "deploying", "healthy", "unhealthy", "failed", "rolled_back"
    started_at: datetime
    completed_at: Optional[datetime] = None
    health_check_status: str = "unknown"
    error_message: Optional[str] = None
    rollback_available: bool = False


class RegionConfigManager:
    """
    Manager for regional configuration and compliance.
    
    Handles region-specific settings, compliance requirements,
    and data residency rules for global deployments.
    """

    def __init__(self):
        """Initialize region configuration manager."""
        self.region_configs = self._initialize_region_configs()

    def _initialize_region_configs(self) -> Dict[DeploymentRegion, RegionConfig]:
        """Initialize default region configurations."""
        configs = {}
        
        # US East (Virginia)
        configs[DeploymentRegion.US_EAST] = RegionConfig(
            region=DeploymentRegion.US_EAST,
            compliance_regimes=[ComplianceRegime.CCPA],
            data_residency_required=False,
            primary_language="en",
            supported_languages=["en", "es"],
            timezone="America/New_York",
            infrastructure_provider=InfrastructureProvider.AWS,
            instance_types={
                ServiceType.FRONTEND: "t3.medium",
                ServiceType.BACKEND_API: "t3.large",
                ServiceType.DATABASE: "r5.xlarge",
                ServiceType.CACHE: "r5.large"
            },
            scaling_limits={"min_instances": 2, "max_instances": 20},
            backup_retention_days=30,
            monitoring_endpoints=["https://monitoring.us-east.lixil.com"]
        )
        
        # EU West (Ireland)
        configs[DeploymentRegion.EU_WEST] = RegionConfig(
            region=DeploymentRegion.EU_WEST,
            compliance_regimes=[ComplianceRegime.GDPR],
            data_residency_required=True,
            primary_language="en",
            supported_languages=["en", "de", "fr", "es", "it", "nl"],
            timezone="Europe/Dublin",
            infrastructure_provider=InfrastructureProvider.AWS,
            instance_types={
                ServiceType.FRONTEND: "t3.medium",
                ServiceType.BACKEND_API: "t3.large",
                ServiceType.DATABASE: "r5.xlarge",
                ServiceType.CACHE: "r5.large"
            },
            scaling_limits={"min_instances": 2, "max_instances": 15},
            backup_retention_days=90,  # GDPR compliance
            monitoring_endpoints=["https://monitoring.eu-west.lixil.com"]
        )
        
        # Asia Pacific (Singapore)
        configs[DeploymentRegion.ASIA_PACIFIC] = RegionConfig(
            region=DeploymentRegion.ASIA_PACIFIC,
            compliance_regimes=[ComplianceRegime.PDPA],
            data_residency_required=True,
            primary_language="en",
            supported_languages=["en", "zh", "ja", "ko", "th"],
            timezone="Asia/Singapore",
            infrastructure_provider=InfrastructureProvider.AWS,
            instance_types={
                ServiceType.FRONTEND: "t3.medium",
                ServiceType.BACKEND_API: "t3.large",
                ServiceType.DATABASE: "r5.xlarge",
                ServiceType.CACHE: "r5.large"
            },
            scaling_limits={"min_instances": 2, "max_instances": 12},
            backup_retention_days=60,
            monitoring_endpoints=["https://monitoring.ap-southeast.lixil.com"]
        )
        
        # Asia Northeast (Tokyo)
        configs[DeploymentRegion.ASIA_NORTHEAST] = RegionConfig(
            region=DeploymentRegion.ASIA_NORTHEAST,
            compliance_regimes=[ComplianceRegime.APPI],
            data_residency_required=True,
            primary_language="ja",
            supported_languages=["ja", "en"],
            timezone="Asia/Tokyo",
            infrastructure_provider=InfrastructureProvider.AWS,
            instance_types={
                ServiceType.FRONTEND: "t3.medium",
                ServiceType.BACKEND_API: "t3.large",
                ServiceType.DATABASE: "r5.xlarge",
                ServiceType.CACHE: "r5.large"
            },
            scaling_limits={"min_instances": 2, "max_instances": 10},
            backup_retention_days=365,  # Japanese data retention requirements
            monitoring_endpoints=["https://monitoring.ap-northeast.lixil.com"]
        )
        
        return configs

    def get_region_config(self, region: DeploymentRegion) -> RegionConfig:
        """Get configuration for a specific region."""
        return self.region_configs.get(region)

    def validate_compliance(self, region: DeploymentRegion, data_types: List[str]) -> Dict[str, Any]:
        """Validate compliance requirements for data types in region."""
        config = self.get_region_config(region)
        if not config:
            return {"valid": False, "error": "Region not configured"}
        
        compliance_checks = {}
        
        for regime in config.compliance_regimes:
            if regime == ComplianceRegime.GDPR:
                compliance_checks["gdpr"] = self._check_gdpr_compliance(data_types, config)
            elif regime == ComplianceRegime.CCPA:
                compliance_checks["ccpa"] = self._check_ccpa_compliance(data_types, config)
            elif regime == ComplianceRegime.PDPA:
                compliance_checks["pdpa"] = self._check_pdpa_compliance(data_types, config)
            elif regime == ComplianceRegime.APPI:
                compliance_checks["appi"] = self._check_appi_compliance(data_types, config)
        
        all_valid = all(check.get("valid", False) for check in compliance_checks.values())
        
        return {
            "valid": all_valid,
            "region": region.value,
            "data_residency_required": config.data_residency_required,
            "compliance_checks": compliance_checks
        }

    def _check_gdpr_compliance(self, data_types: List[str], config: RegionConfig) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        sensitive_data = ["personal_data", "biometric_data", "health_data", "financial_data"]
        has_sensitive = any(dt in sensitive_data for dt in data_types)
        
        return {
            "valid": config.data_residency_required or not has_sensitive,
            "requirements": [
                "Data must remain in EU",
                "Backup retention: 90 days maximum",
                "Right to be forgotten implementation required",
                "Data processing consent tracking required"
            ],
            "sensitive_data_detected": has_sensitive
        }

    def _check_ccpa_compliance(self, data_types: List[str], config: RegionConfig) -> Dict[str, Any]:
        """Check CCPA compliance requirements."""
        personal_data = ["personal_data", "contact_data", "behavioral_data"]
        has_personal = any(dt in personal_data for dt in data_types)
        
        return {
            "valid": True,  # CCPA doesn't require data residency
            "requirements": [
                "Right to know data collection",
                "Right to delete personal information",
                "Right to opt-out of sale",
                "Non-discrimination for privacy rights"
            ],
            "personal_data_detected": has_personal
        }

    def _check_pdpa_compliance(self, data_types: List[str], config: RegionConfig) -> Dict[str, Any]:
        """Check PDPA compliance requirements."""
        return {
            "valid": config.data_residency_required,
            "requirements": [
                "Data localization in Singapore",
                "Consent management required",
                "Data breach notification within 72 hours",
                "Data protection officer appointment"
            ]
        }

    def _check_appi_compliance(self, data_types: List[str], config: RegionConfig) -> Dict[str, Any]:
        """Check APPI compliance requirements."""
        return {
            "valid": config.data_residency_required,
            "requirements": [
                "Personal data protection measures",
                "Cross-border transfer restrictions",
                "Data retention: 365 days maximum",
                "Privacy policy disclosure required"
            ]
        }


class KubernetesDeploymentManager:
    """
    Kubernetes deployment manager for container orchestration.
    
    Handles Kubernetes deployments, services, ingress, and
    auto-scaling configurations across multiple clusters.
    """

    def __init__(self, kubeconfig_path: Optional[str] = None):
        """
        Initialize Kubernetes deployment manager.
        
        Args:
            kubeconfig_path: Path to kubeconfig file (optional)
        """
        self.kubeconfig_path = kubeconfig_path
        self.k8s_client = None
        self.apps_v1 = None
        self.core_v1 = None

    async def initialize(self):
        """Initialize Kubernetes client."""
        try:
            if self.kubeconfig_path:
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                # Try in-cluster config first, then default kubeconfig
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
            
            self.k8s_client = client.ApiClient()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            
            logger.info("Kubernetes client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def generate_deployment_manifest(self, spec: ServiceDeploymentSpec, 
                                   target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        labels = {
            "app": spec.service_name,
            "version": target.version,
            "environment": target.environment.value,
            "region": target.region.value
        }
        
        # Environment variables with region-specific overrides
        env_vars = []
        for key, value in spec.environment_variables.items():
            env_vars.append({"name": key, "value": value})
        
        # Add region-specific environment variables
        env_vars.extend([
            {"name": "DEPLOYMENT_REGION", "value": target.region.value},
            {"name": "DEPLOYMENT_ENVIRONMENT", "value": target.environment.value},
            {"name": "SERVICE_VERSION", "value": target.version}
        ])
        
        # Resource requirements
        resources = {
            "requests": {
                "cpu": spec.resource_requirements.get("cpu_request", "100m"),
                "memory": spec.resource_requirements.get("memory_request", "128Mi")
            },
            "limits": {
                "cpu": spec.resource_requirements.get("cpu_limit", "500m"),
                "memory": spec.resource_requirements.get("memory_limit", "512Mi")
            }
        }
        
        # Container specification
        container = {
            "name": spec.service_name,
            "image": spec.image,
            "ports": [{"containerPort": spec.port}],
            "env": env_vars,
            "resources": resources,
            "livenessProbe": {
                "httpGet": {
                    "path": spec.health_check_path,
                    "port": spec.port
                },
                "initialDelaySeconds": 30,
                "periodSeconds": 10
            },
            "readinessProbe": {
                "httpGet": {
                    "path": spec.health_check_path,
                    "port": spec.port
                },
                "initialDelaySeconds": 5,
                "periodSeconds": 5
            }
        }
        
        # Add volume mounts if persistent storage is required
        if spec.persistent_storage:
            container["volumeMounts"] = [{
                "name": f"{spec.service_name}-storage",
                "mountPath": "/data"
            }]
        
        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{spec.service_name}-{target.environment.value}",
                "namespace": f"lixil-ai-hub-{target.environment.value}",
                "labels": labels
            },
            "spec": {
                "replicas": spec.replicas,
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "containers": [container],
                        "restartPolicy": "Always"
                    }
                }
            }
        }
        
        # Add persistent volume if required
        if spec.persistent_storage:
            deployment["spec"]["template"]["spec"]["volumes"] = [{
                "name": f"{spec.service_name}-storage",
                "persistentVolumeClaim": {
                    "claimName": f"{spec.service_name}-pvc-{target.environment.value}"
                }
            }]
        
        return deployment

    def generate_service_manifest(self, spec: ServiceDeploymentSpec, 
                                target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        labels = {
            "app": spec.service_name,
            "environment": target.environment.value,
            "region": target.region.value
        }
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{spec.service_name}-service-{target.environment.value}",
                "namespace": f"lixil-ai-hub-{target.environment.value}",
                "labels": labels
            },
            "spec": {
                "selector": labels,
                "ports": [{
                    "port": 80,
                    "targetPort": spec.port,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
        
        return service

    def generate_hpa_manifest(self, spec: ServiceDeploymentSpec, 
                            target: DeploymentTarget) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        if not spec.auto_scaling:
            return None
        
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{spec.service_name}-hpa-{target.environment.value}",
                "namespace": f"lixil-ai-hub-{target.environment.value}"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{spec.service_name}-{target.environment.value}"
                },
                "minReplicas": max(1, spec.replicas // 2),
                "maxReplicas": spec.replicas * 3,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        return hpa

    async def deploy_service(self, spec: ServiceDeploymentSpec, 
                           target: DeploymentTarget) -> bool:
        """Deploy service to Kubernetes cluster."""
        try:
            namespace = f"lixil-ai-hub-{target.environment.value}"
            
            # Ensure namespace exists
            await self._ensure_namespace(namespace)
            
            # Generate manifests
            deployment_manifest = self.generate_deployment_manifest(spec, target)
            service_manifest = self.generate_service_manifest(spec, target)
            hpa_manifest = self.generate_hpa_manifest(spec, target)
            
            # Apply deployment
            try:
                self.apps_v1.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment_manifest
                )
                logger.info(f"Created deployment: {deployment_manifest['metadata']['name']}")
            except client.exceptions.ApiException as e:
                if e.status == 409:  # Already exists, update instead
                    self.apps_v1.patch_namespaced_deployment(
                        name=deployment_manifest['metadata']['name'],
                        namespace=namespace,
                        body=deployment_manifest
                    )
                    logger.info(f"Updated deployment: {deployment_manifest['metadata']['name']}")
                else:
                    raise
            
            # Apply service
            try:
                self.core_v1.create_namespaced_service(
                    namespace=namespace,
                    body=service_manifest
                )
                logger.info(f"Created service: {service_manifest['metadata']['name']}")
            except client.exceptions.ApiException as e:
                if e.status == 409:  # Already exists, update instead
                    self.core_v1.patch_namespaced_service(
                        name=service_manifest['metadata']['name'],
                        namespace=namespace,
                        body=service_manifest
                    )
                    logger.info(f"Updated service: {service_manifest['metadata']['name']}")
                else:
                    raise
            
            # Apply HPA if auto-scaling is enabled
            if hpa_manifest:
                autoscaling_v2 = client.AutoscalingV2Api()
                try:
                    autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                        namespace=namespace,
                        body=hpa_manifest
                    )
                    logger.info(f"Created HPA: {hpa_manifest['metadata']['name']}")
                except client.exceptions.ApiException as e:
                    if e.status == 409:  # Already exists, update instead
                        autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                            name=hpa_manifest['metadata']['name'],
                            namespace=namespace,
                            body=hpa_manifest
                        )
                        logger.info(f"Updated HPA: {hpa_manifest['metadata']['name']}")
                    else:
                        raise
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {spec.service_name}: {e}")
            return False

    async def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists, create if it doesn't."""
        try:
            self.core_v1.read_namespace(name=namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Namespace doesn't exist, create it
                namespace_manifest = {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {
                        "name": namespace,
                        "labels": {
                            "project": "lixil-ai-hub",
                            "managed-by": "deployment-manager"
                        }
                    }
                }
                self.core_v1.create_namespace(body=namespace_manifest)
                logger.info(f"Created namespace: {namespace}")
            else:
                raise

    async def check_deployment_health(self, service_name: str, target: DeploymentTarget) -> Dict[str, Any]:
        """Check health of a deployment."""
        namespace = f"lixil-ai-hub-{target.environment.value}"
        deployment_name = f"{service_name}-{target.environment.value}"
        
        try:
            # Get deployment status
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Get pod status
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={service_name}"
            )
            
            ready_replicas = deployment.status.ready_replicas or 0
            desired_replicas = deployment.spec.replicas
            
            pod_statuses = []
            for pod in pods.items:
                pod_statuses.append({
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "ready": all(condition.status == "True" 
                               for condition in pod.status.conditions or []
                               if condition.type == "Ready")
                })
            
            health_status = {
                "healthy": ready_replicas == desired_replicas,
                "ready_replicas": ready_replicas,
                "desired_replicas": desired_replicas,
                "pod_statuses": pod_statuses,
                "last_updated": deployment.status.conditions[-1].last_update_time.isoformat() if deployment.status.conditions else None
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to check deployment health: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }


class GlobalDeploymentManager:
    """
    Main manager for global deployment orchestration.
    
    Coordinates multi-region deployments, compliance validation,
    and infrastructure management across the LIXIL AI Hub Platform.
    """

    def __init__(self, database_url: str, config: GlobalDeploymentConfig):
        """
        Initialize global deployment manager.
        
        Args:
            database_url: PostgreSQL database connection URL
            config: Global deployment configuration
        """
        self.database_url = database_url
        self.db_pool = None
        self.config = config
        self.region_manager = RegionConfigManager()
        self.k8s_managers: Dict[DeploymentRegion, KubernetesDeploymentManager] = {}
        self.active_deployments: Dict[str, DeploymentStatus] = {}

    async def initialize(self):
        """Initialize deployment manager and database."""
        # Initialize database
        self.db_pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        await self._create_schema()
        
        # Initialize Kubernetes managers for each region
        for region in self.config.regions:
            k8s_manager = KubernetesDeploymentManager()
            await k8s_manager.initialize()
            self.k8s_managers[region] = k8s_manager
        
        logger.info("Global deployment manager initialized")

    async def close(self):
        """Close all connections."""
        if self.db_pool:
            await self.db_pool.close()

    async def _create_schema(self):
        """Create database schema for deployment tracking."""
        schema_sql = """
        -- Deployments table
        CREATE TABLE IF NOT EXISTS deployments (
            deployment_id VARCHAR(64) PRIMARY KEY,
            service_name VARCHAR(100) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            region VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE NOT NULL,
            completed_at TIMESTAMP WITH TIME ZONE,
            health_check_url TEXT,
            error_message TEXT,
            rollback_version VARCHAR(50),
            deployment_config JSONB,
            created_by VARCHAR(100),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
        );

        -- Deployment history table
        CREATE TABLE IF NOT EXISTS deployment_history (
            history_id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(64) NOT NULL,
            previous_status VARCHAR(50),
            new_status VARCHAR(50) NOT NULL,
            changed_at TIMESTAMP WITH TIME ZONE NOT NULL,
            changed_by VARCHAR(100),
            notes TEXT,
            FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
        );

        -- Health checks table
        CREATE TABLE IF NOT EXISTS health_checks (
            check_id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(64) NOT NULL,
            check_type VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            response_time_ms INTEGER,
            error_message TEXT,
            checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
            FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_deployments_service ON deployments(service_name);
        CREATE INDEX IF NOT EXISTS idx_deployments_environment ON deployments(environment);
        CREATE INDEX IF NOT EXISTS idx_deployments_region ON deployments(region);
        CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
        CREATE INDEX IF NOT EXISTS idx_deployment_history_deployment ON deployment_history(deployment_id);
        CREATE INDEX IF NOT EXISTS idx_health_checks_deployment ON health_checks(deployment_id);
        CREATE INDEX IF NOT EXISTS idx_health_checks_checked_at ON health_checks(checked_at);
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def deploy_globally(self, spec: ServiceDeploymentSpec, 
                            targets: List[DeploymentTarget]) -> Dict[str, DeploymentStatus]:
        """Deploy service to multiple regions."""
        deployment_results = {}
        
        for target in targets:
            # Validate compliance for target region
            compliance_result = self.region_manager.validate_compliance(
                target.region, 
                ["user_data", "system_logs"]  # Example data types
            )
            
            if not compliance_result["valid"]:
                logger.error(f"Compliance validation failed for {target.region}: {compliance_result}")
                deployment_results[target.region.value] = DeploymentStatus(
                    deployment_id=f"deploy-{target.region.value}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    target=target,
                    status="failed",
                    started_at=datetime.now(),
                    error_message=f"Compliance validation failed: {compliance_result}"
                )
                continue
            
            # Start deployment
            deployment_id = await self._start_deployment(spec, target)
            
            try:
                # Deploy to Kubernetes
                k8s_manager = self.k8s_managers.get(target.region)
                if not k8s_manager:
                    raise Exception(f"No Kubernetes manager for region {target.region}")
                
                success = await k8s_manager.deploy_service(spec, target)
                
                if success:
                    # Update deployment status
                    await self._update_deployment_status(deployment_id, "healthy")
                    
                    # Perform health check
                    health_status = await self._perform_health_check(deployment_id, target)
                    
                    deployment_results[target.region.value] = DeploymentStatus(
                        deployment_id=deployment_id,
                        target=target,
                        status="healthy" if health_status["healthy"] else "unhealthy",
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                        health_check_status="passed" if health_status["healthy"] else "failed"
                    )
                else:
                    await self._update_deployment_status(deployment_id, "failed")
                    deployment_results[target.region.value] = DeploymentStatus(
                        deployment_id=deployment_id,
                        target=target,
                        status="failed",
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                        error_message="Kubernetes deployment failed"
                    )
                    
            except Exception as e:
                logger.error(f"Deployment failed for {target.region}: {e}")
                await self._update_deployment_status(deployment_id, "failed", str(e))
                deployment_results[target.region.value] = DeploymentStatus(
                    deployment_id=deployment_id,
                    target=target,
                    status="failed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error_message=str(e)
                )
        
        return deployment_results

    async def _start_deployment(self, spec: ServiceDeploymentSpec, target: DeploymentTarget) -> str:
        """Start a new deployment and record it in database."""
        deployment_id = f"deploy-{target.region.value}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        now = datetime.now()
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO deployments (
                    deployment_id, service_name, version, environment, region,
                    status, started_at, health_check_url, deployment_config,
                    created_by, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                deployment_id, spec.service_name, target.version,
                target.environment.value, target.region.value, "deploying",
                now, target.health_check_url, json.dumps(spec.dict()),
                "deployment-manager", now
            )
        
        return deployment_id

    async def _update_deployment_status(self, deployment_id: str, status: str, error_message: Optional[str] = None):
        """Update deployment status in database."""
        now = datetime.now()
        
        async with self.db_pool.acquire() as conn:
            # Get current status for history
            current = await conn.fetchrow(
                "SELECT status FROM deployments WHERE deployment_id = $1",
                deployment_id
            )
            
            # Update deployment
            if error_message:
                await conn.execute("""
                    UPDATE deployments 
                    SET status = $1, error_message = $2, updated_at = $3, completed_at = $4
                    WHERE deployment_id = $5
                """, status, error_message, now, now, deployment_id)
            else:
                await conn.execute("""
                    UPDATE deployments 
                    SET status = $1, updated_at = $2, completed_at = $3
                    WHERE deployment_id = $4
                """, status, now, now, deployment_id)
            
            # Add to history
            if current:
                await conn.execute("""
                    INSERT INTO deployment_history (
                        deployment_id, previous_status, new_status, changed_at, changed_by
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    deployment_id, current["status"], status, now, "deployment-manager"
                )

    async def _perform_health_check(self, deployment_id: str, target: DeploymentTarget) -> Dict[str, Any]:
        """Perform health check on deployed service."""
        if not target.health_check_url:
            return {"healthy": True, "message": "No health check URL configured"}
        
        try:
            start_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(target.health_check_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    healthy = response.status == 200
                    
                    # Log health check
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO health_checks (
                                deployment_id, check_type, status, response_time_ms, checked_at
                            ) VALUES ($1, $2, $3, $4, $5)
                        """,
                            deployment_id, "http", "passed" if healthy else "failed",
                            int(response_time), datetime.now()
                        )
                    
                    return {
                        "healthy": healthy,
                        "response_time_ms": response_time,
                        "status_code": response.status
                    }
                    
        except Exception as e:
            # Log failed health check
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO health_checks (
                        deployment_id, check_type, status, error_message, checked_at
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    deployment_id, "http", "failed", str(e), datetime.now()
                )
            
            return {
                "healthy": False,
                "error": str(e)
            }

    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get status of a specific deployment."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM deployments WHERE deployment_id = $1
            """, deployment_id)
            
            if row:
                target = DeploymentTarget(
                    environment=DeploymentEnvironment(row["environment"]),
                    region=DeploymentRegion(row["region"]),
                    version=row["version"],
                    health_check_url=row["health_check_url"]
                )
                
                return DeploymentStatus(
                    deployment_id=row["deployment_id"],
                    target=target,
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    error_message=row["error_message"],
                    rollback_available=bool(row["rollback_version"])
                )
        
        return None

    async def get_global_deployment_overview(self) -> Dict[str, Any]:
        """Get overview of all deployments across regions."""
        async with self.db_pool.acquire() as conn:
            # Get deployment counts by status and region
            status_counts = await conn.fetch("""
                SELECT region, status, COUNT(*) as count
                FROM deployments
                WHERE started_at >= NOW() - INTERVAL '7 days'
                GROUP BY region, status
                ORDER BY region, status
            """)
            
            # Get recent deployments
            recent_deployments = await conn.fetch("""
                SELECT deployment_id, service_name, version, environment, region, status, started_at
                FROM deployments
                ORDER BY started_at DESC
                LIMIT 20
            """)
            
            # Get health check summary
            health_summary = await conn.fetch("""
                SELECT 
                    d.region,
                    COUNT(CASE WHEN hc.status = 'passed' THEN 1 END) as healthy_checks,
                    COUNT(CASE WHEN hc.status = 'failed' THEN 1 END) as failed_checks,
                    AVG(hc.response_time_ms) as avg_response_time
                FROM deployments d
                LEFT JOIN health_checks hc ON d.deployment_id = hc.deployment_id
                WHERE hc.checked_at >= NOW() - INTERVAL '1 hour'
                GROUP BY d.region
            """)
        
        return {
            "status_by_region": [dict(row) for row in status_counts],
            "recent_deployments": [dict(row) for row in recent_deployments],
            "health_summary": [dict(row) for row in health_summary],
            "total_regions": len(self.config.regions),
            "total_environments": len(self.config.environments)
        }

