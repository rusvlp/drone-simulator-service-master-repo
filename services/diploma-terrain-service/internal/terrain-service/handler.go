package terrain_service

import (
	"context"

	"github.com/TwiLightDM/diploma-terrain-service/proto/terrainservicepb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Handler struct {
	terrainservicepb.UnimplementedTerrainServiceServer
	service Service
}

func NewHandler(service Service) *Handler {
	return &Handler{service: service}
}

func (h *Handler) SubmitJob(ctx context.Context, req *terrainservicepb.SubmitJobRequest) (*terrainservicepb.SubmitJobResponse, error) {
	jobID, err := h.service.SubmitJob(ctx, req.UserId, req.Image, req.ScaleZ, req.YUp, req.TextureMode)
	if err != nil {
		if err.Error() == "daily generation limit reached" {
			return nil, status.Error(codes.ResourceExhausted, err.Error())
		}
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &terrainservicepb.SubmitJobResponse{JobId: jobID}, nil
}

func (h *Handler) GetJobStatus(ctx context.Context, req *terrainservicepb.GetJobStatusRequest) (*terrainservicepb.GetJobStatusResponse, error) {
	job, err := h.service.GetJobStatus(ctx, req.JobId)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	if job == nil {
		return nil, status.Error(codes.NotFound, "job not found")
	}
	return &terrainservicepb.GetJobStatusResponse{
		JobId:      job.ID,
		Status:     job.Status,
		ModelUrl:   job.ModelURL,
		TextureUrl: job.TextureURL,
		TreesUrl:   job.TreesURL,
		Error:      job.Error,
		CreatedAt:  timestamppb.New(job.CreatedAt),
	}, nil
}

func (h *Handler) GetUserJobs(ctx context.Context, req *terrainservicepb.GetUserJobsRequest) (*terrainservicepb.GetUserJobsResponse, error) {
	jobs, err := h.service.GetUserJobs(ctx, req.UserId)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	var pbJobs []*terrainservicepb.GetJobStatusResponse
	for _, j := range jobs {
		pbJobs = append(pbJobs, &terrainservicepb.GetJobStatusResponse{
			JobId:      j.ID,
			Status:     j.Status,
			ModelUrl:   j.ModelURL,
			TextureUrl: j.TextureURL,
			TreesUrl:   j.TreesURL,
			Error:      j.Error,
			CreatedAt:  timestamppb.New(j.CreatedAt),
		})
	}
	return &terrainservicepb.GetUserJobsResponse{Jobs: pbJobs}, nil
}

func (h *Handler) GetLimit(ctx context.Context, req *terrainservicepb.GetLimitRequest) (*terrainservicepb.GetLimitResponse, error) {
	limit, err := h.service.GetLimit(ctx, req.UserId)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &terrainservicepb.GetLimitResponse{
		UserId:     limit.UserID,
		DailyLimit: int32(limit.DailyLimit),
		UsedToday:  int32(limit.UsedToday),
		ResetAt:    timestamppb.New(limit.ResetAt),
	}, nil
}

func (h *Handler) SetLimit(ctx context.Context, req *terrainservicepb.SetLimitRequest) (*terrainservicepb.SetLimitResponse, error) {
	limit, err := h.service.SetLimit(ctx, req.UserId, int(req.DailyLimit))
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &terrainservicepb.SetLimitResponse{
		UserId:     limit.UserID,
		DailyLimit: int32(limit.DailyLimit),
	}, nil
}
