/**
 * LIXIL AI Hub Admin Dashboard
 * 
 * Main dashboard component for AI Council members to manage policies,
 * permanent statements, and system administration.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Badge
} from '@mui/material';
import {
  Upload,
  Add,
  Edit,
  Delete,
  Visibility,
  CheckCircle,
  Cancel,
  Analytics,
  People,
  Description,
  QuestionAnswer,
  Refresh
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const AdminDashboard = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [policies, setPolicies] = useState([]);
  const [statements, setStatements] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [selectedPolicy, setSelectedPolicy] = useState(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [statementDialogOpen, setStatementDialogOpen] = useState(false);
  const [approvalDialogOpen, setApprovalDialogOpen] = useState(false);
  const [selectedContent, setSelectedContent] = useState(null);

  // Form states
  const [policyForm, setPolicyForm] = useState({
    title: '',
    category: '',
    language: 'en',
    region: '',
    tags: '',
    file: null
  });

  const [statementForm, setStatementForm] = useState({
    question: '',
    answer: '',
    category: '',
    priority: 5,
    language: 'en',
    region: '',
    tags: ''
  });

  const [searchForm, setSearchForm] = useState({
    query: '',
    category: '',
    status: '',
    language: '',
    region: ''
  });

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadPolicies(),
        loadStatements(),
        loadAnalytics()
      ]);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPolicies = async () => {
    try {
      const response = await fetch('/api/admin/policies/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          limit: 50,
          offset: 0,
          ...searchForm
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setPolicies(data.policies);
      }
    } catch (error) {
      console.error('Failed to load policies:', error);
    }
  };

  const loadStatements = async () => {
    try {
      const response = await fetch('/api/admin/statements', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setStatements(data.statements || []);
      }
    } catch (error) {
      console.error('Failed to load statements:', error);
    }
  };

  const loadAnalytics = async () => {
    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - 30);

      const response = await fetch('/api/admin/analytics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          start_date: startDate.toISOString(),
          end_date: endDate.toISOString(),
          granularity: 'day'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data);
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const handlePolicyUpload = async () => {
    if (!policyForm.file || !policyForm.title || !policyForm.category) {
      alert('Please fill in all required fields and select a file');
      return;
    }

    const formData = new FormData();
    formData.append('file', policyForm.file);
    formData.append('title', policyForm.title);
    formData.append('category', policyForm.category);
    formData.append('language', policyForm.language);
    formData.append('region', policyForm.region);
    formData.append('tags', JSON.stringify(policyForm.tags.split(',').map(t => t.trim()).filter(t => t)));

    try {
      const response = await fetch('/api/admin/policies/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: formData
      });

      if (response.ok) {
        setUploadDialogOpen(false);
        setPolicyForm({
          title: '',
          category: '',
          language: 'en',
          region: '',
          tags: '',
          file: null
        });
        loadPolicies();
        alert('Policy uploaded successfully');
      } else {
        const error = await response.json();
        alert(`Upload failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed');
    }
  };

  const handleStatementCreate = async () => {
    if (!statementForm.question || !statementForm.answer || !statementForm.category) {
      alert('Please fill in all required fields');
      return;
    }

    try {
      const response = await fetch('/api/admin/statements', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          ...statementForm,
          tags: statementForm.tags.split(',').map(t => t.trim()).filter(t => t)
        })
      });

      if (response.ok) {
        setStatementDialogOpen(false);
        setStatementForm({
          question: '',
          answer: '',
          category: '',
          priority: 5,
          language: 'en',
          region: '',
          tags: ''
        });
        loadStatements();
        alert('Statement created successfully');
      } else {
        const error = await response.json();
        alert(`Creation failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Creation failed:', error);
      alert('Creation failed');
    }
  };

  const handleContentApproval = async (action, comments = '') => {
    if (!selectedContent) return;

    try {
      const response = await fetch('/api/admin/content/approve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          content_id: selectedContent.id,
          content_type: selectedContent.type,
          action: action,
          comments: comments
        })
      });

      if (response.ok) {
        setApprovalDialogOpen(false);
        setSelectedContent(null);
        loadPolicies();
        loadStatements();
        alert(`Content ${action} successfully`);
      } else {
        const error = await response.json();
        alert(`Action failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Approval failed:', error);
      alert('Action failed');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'published': return 'success';
      case 'approved': return 'info';
      case 'pending_review': return 'warning';
      case 'rejected': return 'error';
      case 'draft': return 'default';
      default: return 'default';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString();
  };

  const DashboardOverview = () => (
    <Grid container spacing={3}>
      {/* Summary Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center">
              <Description color="primary" sx={{ mr: 2 }} />
              <Box>
                <Typography variant="h4">
                  {analytics?.policy_statistics?.total_policies || 0}
                </Typography>
                <Typography color="textSecondary">Total Policies</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center">
              <QuestionAnswer color="secondary" sx={{ mr: 2 }} />
              <Box>
                <Typography variant="h4">
                  {analytics?.statement_statistics?.total_statements || 0}
                </Typography>
                <Typography color="textSecondary">Statements</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center">
              <CheckCircle color="success" sx={{ mr: 2 }} />
              <Box>
                <Typography variant="h4">
                  {analytics?.policy_statistics?.published_policies || 0}
                </Typography>
                <Typography color="textSecondary">Published</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center">
              <Badge badgeContent={analytics?.policy_statistics?.pending_policies || 0} color="warning">
                <Analytics color="info" sx={{ mr: 2 }} />
              </Badge>
              <Box>
                <Typography variant="h4">
                  {analytics?.policy_statistics?.pending_policies || 0}
                </Typography>
                <Typography color="textSecondary">Pending Review</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Charts */}
      {analytics?.time_series && (
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Policy Creation Trend</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analytics.time_series}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="policy_count" stroke="#8884d8" />
                  <Line type="monotone" dataKey="published_count" stroke="#82ca9d" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      )}

      {analytics?.approval_activity && (
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Approval Activity</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics.approval_activity}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="action" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      )}
    </Grid>
  );

  const PolicyManagement = () => (
    <Box>
      <Box display="flex" justifyContent="between" alignItems="center" mb={3}>
        <Typography variant="h5">Policy Management</Typography>
        <Button
          variant="contained"
          startIcon={<Upload />}
          onClick={() => setUploadDialogOpen(true)}
        >
          Upload Policy
        </Button>
      </Box>

      {/* Search Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Search"
                value={searchForm.query}
                onChange={(e) => setSearchForm({...searchForm, query: e.target.value})}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select
                  value={searchForm.category}
                  onChange={(e) => setSearchForm({...searchForm, category: e.target.value})}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="AI Policy">AI Policy</MenuItem>
                  <MenuItem value="Data Privacy">Data Privacy</MenuItem>
                  <MenuItem value="Usage Guidelines">Usage Guidelines</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={searchForm.status}
                  onChange={(e) => setSearchForm({...searchForm, status: e.target.value})}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="draft">Draft</MenuItem>
                  <MenuItem value="pending_review">Pending Review</MenuItem>
                  <MenuItem value="approved">Approved</MenuItem>
                  <MenuItem value="published">Published</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <Button variant="outlined" onClick={loadPolicies} startIcon={<Refresh />}>
                Search
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Policies Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Title</TableCell>
              <TableCell>Category</TableCell>
              <TableCell>Version</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Language</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {policies.map((policy) => (
              <TableRow key={policy.policy_id}>
                <TableCell>{policy.title}</TableCell>
                <TableCell>{policy.category}</TableCell>
                <TableCell>{policy.version}</TableCell>
                <TableCell>
                  <Chip 
                    label={policy.status} 
                    color={getStatusColor(policy.status)}
                    size="small"
                  />
                </TableCell>
                <TableCell>{policy.language}</TableCell>
                <TableCell>{formatDate(policy.created_at)}</TableCell>
                <TableCell>
                  <IconButton 
                    size="small"
                    onClick={() => setSelectedPolicy(policy)}
                  >
                    <Visibility />
                  </IconButton>
                  {policy.status === 'pending_review' && (
                    <IconButton 
                      size="small"
                      onClick={() => {
                        setSelectedContent({
                          id: policy.policy_id,
                          type: 'policy_document',
                          title: policy.title
                        });
                        setApprovalDialogOpen(true);
                      }}
                    >
                      <CheckCircle />
                    </IconButton>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );

  const StatementManagement = () => (
    <Box>
      <Box display="flex" justifyContent="between" alignItems="center" mb={3}>
        <Typography variant="h5">Permanent Statements</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setStatementDialogOpen(true)}
        >
          Create Statement
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Question</TableCell>
              <TableCell>Category</TableCell>
              <TableCell>Priority</TableCell>
              <TableCell>Language</TableCell>
              <TableCell>Usage Count</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {statements.map((statement) => (
              <TableRow key={statement.statement_id}>
                <TableCell>{statement.question.substring(0, 100)}...</TableCell>
                <TableCell>{statement.category}</TableCell>
                <TableCell>{statement.priority}</TableCell>
                <TableCell>{statement.language}</TableCell>
                <TableCell>{statement.usage_count}</TableCell>
                <TableCell>
                  <Chip 
                    label={statement.is_active ? 'Active' : 'Inactive'} 
                    color={statement.is_active ? 'success' : 'default'}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <IconButton size="small">
                    <Edit />
                  </IconButton>
                  <IconButton size="small">
                    <Delete />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        AI Hub Admin Portal
      </Typography>

      <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Dashboard" />
        <Tab label="Policies" />
        <Tab label="Statements" />
        <Tab label="Users" />
        <Tab label="Analytics" />
      </Tabs>

      {loading && (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      )}

      {!loading && (
        <>
          {activeTab === 0 && <DashboardOverview />}
          {activeTab === 1 && <PolicyManagement />}
          {activeTab === 2 && <StatementManagement />}
          {activeTab === 3 && <Typography>User Management (Coming Soon)</Typography>}
          {activeTab === 4 && <Typography>Advanced Analytics (Coming Soon)</Typography>}
        </>
      )}

      {/* Policy Upload Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Upload Policy Document</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Title"
                value={policyForm.title}
                onChange={(e) => setPolicyForm({...policyForm, title: e.target.value})}
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Category"
                value={policyForm.category}
                onChange={(e) => setPolicyForm({...policyForm, category: e.target.value})}
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select
                  value={policyForm.language}
                  onChange={(e) => setPolicyForm({...policyForm, language: e.target.value})}
                >
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="ja">Japanese</MenuItem>
                  <MenuItem value="de">German</MenuItem>
                  <MenuItem value="fr">French</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Region (Optional)"
                value={policyForm.region}
                onChange={(e) => setPolicyForm({...policyForm, region: e.target.value})}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Tags (comma-separated)"
                value={policyForm.tags}
                onChange={(e) => setPolicyForm({...policyForm, tags: e.target.value})}
              />
            </Grid>
            <Grid item xs={12}>
              <input
                type="file"
                accept=".pdf,.docx,.doc,.txt,.md"
                onChange={(e) => setPolicyForm({...policyForm, file: e.target.files[0]})}
                style={{ width: '100%', padding: '10px' }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
          <Button onClick={handlePolicyUpload} variant="contained">Upload</Button>
        </DialogActions>
      </Dialog>

      {/* Statement Creation Dialog */}
      <Dialog open={statementDialogOpen} onClose={() => setStatementDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Permanent Statement</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Question"
                multiline
                rows={2}
                value={statementForm.question}
                onChange={(e) => setStatementForm({...statementForm, question: e.target.value})}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Answer"
                multiline
                rows={4}
                value={statementForm.answer}
                onChange={(e) => setStatementForm({...statementForm, answer: e.target.value})}
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Category"
                value={statementForm.category}
                onChange={(e) => setStatementForm({...statementForm, category: e.target.value})}
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Priority (1-10)"
                value={statementForm.priority}
                onChange={(e) => setStatementForm({...statementForm, priority: parseInt(e.target.value)})}
                inputProps={{ min: 1, max: 10 }}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStatementDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleStatementCreate} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>

      {/* Content Approval Dialog */}
      <Dialog open={approvalDialogOpen} onClose={() => setApprovalDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Content Approval</DialogTitle>
        <DialogContent>
          {selectedContent && (
            <Box>
              <Typography variant="body1" gutterBottom>
                {selectedContent.title}
              </Typography>
              <TextField
                fullWidth
                label="Comments (Optional)"
                multiline
                rows={3}
                sx={{ mt: 2 }}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApprovalDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => handleContentApproval('reject')} color="error">
            Reject
          </Button>
          <Button onClick={() => handleContentApproval('approve')} color="success" variant="contained">
            Approve
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AdminDashboard;

